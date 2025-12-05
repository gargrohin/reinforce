"""
REINFORCE trainer for sentiment-controlled text generation.
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import GPT2Tokenizer
from pathlib import Path

from reinforce.training.reward import SentimentRewardFunction
from reinforce.training.baseline import RewardBaseline

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


class ReinforceTrainer:
    """
    Trainer for REINFORCE algorithm on sentiment-controlled generation.
    """

    def __init__(
        self,
        model,
        dataloader,
        config,
        device='cuda',
        use_wandb=False
    ):
        """
        Args:
            model: Transformer model with generate_with_log_probs method
            dataloader: DataLoader for training data
            config: Configuration object with hyperparameters
            device: Device to train on
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Initialize components
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.reward_fn = SentimentRewardFunction(
            model_name=config.sentiment_model,
            device=0 if device == 'cuda' else -1
        )

        self.baseline = RewardBaseline(decay=config.baseline_decay)

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tracking
        self.global_step = 0
        self.epoch_metrics = []
        self.best_reward = float('-inf')  # Track best reward for checkpointing

        # Initialize WandB
        if self.use_wandb:
            wandb.init(
                project="reinforce-sentiment",
                config=vars(config),
                name=f"{config.model_name}_lr{config.learning_rate}"
            )

    def train_step(self, batch):
        """
        Single training step.

        Args:
            batch: Batch from dataloader

        Returns:
            Dict with loss, rewards, and other metrics
        """
        prompt_tokens = batch['prompt_tokens'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        target_sentiments = batch['target_sentiment']
        sentence_fragments = batch['sentence_fragment']

        # Generate completions with log probs
        generated_tokens, log_probs = self.model.generate_with_log_probs(
            prompt_tokens,
            max_new_tokens=self.config.max_new_tokens,
            attention_mask=attention_mask
        )

        # Decode generated tokens to text
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Compute rewards
        rewards = self.reward_fn(
            sentence_fragments,
            generated_texts,
            target_sentiments
        ).to(self.device)

        # Update baseline
        self.baseline.update(rewards)
        baseline_value = self.baseline.get()

        # REINFORCE loss: -log_prob * (reward - baseline)
        # log_probs shape: (batch_size, max_new_tokens)
        # rewards shape: (batch_size,)

        # Sum log probs across generated tokens
        log_probs_sum = log_probs.sum(dim=1)  # (batch_size,)

        # Compute advantages (reward - baseline)
        advantages = rewards - baseline_value

        # REINFORCE loss (negative because we want to maximize)
        loss = -(log_probs_sum * advantages).mean()

        # Backprop and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log to WandB
        if self.use_wandb:
            wandb.log({
                'train/loss': loss.item(),
                'train/reward_mean': rewards.mean().item(),
                'train/reward_std': rewards.std().item(),
                'train/baseline': baseline_value,
                'train/advantages_mean': advantages.mean().item(),
                'train/log_probs_mean': log_probs_sum.mean().item(),
            }, step=self.global_step)

        return {
            'loss': loss.item(),
            'avg_reward': rewards.mean().item(),
            'rewards': rewards,  # Keep for batch logging
            'generated_texts': generated_texts,  # Keep for batch logging
        }

    def train_epoch(self, epoch):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dict with epoch-level metrics
        """
        self.model.train()

        epoch_losses = []
        epoch_rewards = []

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Train step
            metrics = self.train_step(batch)

            epoch_losses.append(metrics['loss'])
            epoch_rewards.append(metrics['avg_reward'])

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'reward': f"{metrics['avg_reward']:.4f}",
                'baseline': f"{self.baseline.get():.4f}"
            })

            # Log every N steps
            if batch_idx % self.config.log_interval == 0:
                self._log_batch(batch_idx, metrics, batch)

            self.global_step += 1

        # Epoch summary
        epoch_summary = {
            'epoch': epoch,
            'avg_loss': sum(epoch_losses) / len(epoch_losses),
            'avg_reward': sum(epoch_rewards) / len(epoch_rewards),
            'baseline': self.baseline.get()
        }

        return epoch_summary

    def train(self):
        """
        Main training loop.
        """
        print("="*60)
        print("Starting REINFORCE Training")
        print("="*60)
        print(f"Model: {self.config.model_name}")
        print(f"Dataset size: {len(self.dataloader.dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Epochs: {self.config.num_epochs}")
        print("="*60)

        for epoch in range(self.config.num_epochs):
            epoch_summary = self.train_epoch(epoch)
            self.epoch_metrics.append(epoch_summary)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Avg Loss: {epoch_summary['avg_loss']:.4f}")
            print(f"  Avg Reward: {epoch_summary['avg_reward']:.4f}")
            print(f"  Baseline: {epoch_summary['baseline']:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(epoch)

        print("\nTraining completed!")
        return self.epoch_metrics

    def _log_batch(self, batch_idx, metrics, batch):
        """Log batch-level information."""
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx} | Step {self.global_step}")
        print(f"{'='*60}")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Avg Reward: {metrics['avg_reward']:.4f}")
        print(f"Baseline: {self.baseline.get():.4f}")

        # Show 3 examples
        print(f"\nExample Generations:")
        num_examples = min(3, len(batch['prompt_text']))
        for i in range(num_examples):
            prompt = batch['prompt_text'][i]
            generated = metrics['generated_texts'][i]
            target = batch['target_sentiment'][i]
            reward = metrics['rewards'][i].item()

            print(f"\n  Example {i+1}:")
            print(f"    Prompt: {prompt}")
            print(f"    Generated: {generated}")
            print(f"    Target: {target}, Reward: {reward:+.1f}")
        print(f"{'='*60}\n")

    def _save_checkpoint(self, epoch):
        """Save best model checkpoint based on reward."""
        current_reward = self.epoch_metrics[-1]['avg_reward']

        # Only save if this is the best reward so far
        if current_reward > self.best_reward:
            self.best_reward = current_reward

            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / "best_model.pt"

            checkpoint = {
                'epoch': epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'baseline_value': self.baseline.get(),
                'best_reward': self.best_reward,
                'config': vars(self.config),
                'epoch_metrics': self.epoch_metrics,
            }

            torch.save(checkpoint, checkpoint_path)
            print(f"✓ New best model saved! Reward: {current_reward:.4f} -> {checkpoint_path}")

            # Log to WandB
            if self.use_wandb:
                wandb.save(str(checkpoint_path))
