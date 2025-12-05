"""
Training script for sentiment-controlled text generation with REINFORCE.

Usage:
    python scripts/train_sentiment_reinforce.py
    python scripts/train_sentiment_reinforce.py --epochs 5 --batch_size 32
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from reinforce.models.models import Transformer
from reinforce.dataloader.sentiment_dataset import create_sentiment_dataloader
from reinforce.training.reinforce_trainer import ReinforceTrainer
from reinforce.configs.sentiment_config import SentimentReinforceConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train REINFORCE on sentiment task")

    # Model
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model size: gpt2, gpt2-medium, gpt2-large, gpt2-xl")

    # Data
    parser.add_argument("--dataset", type=str, default="imdb",
                        help="Dataset name from HuggingFace")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to use (None = all)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")

    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--max_new_tokens", type=int, default=10,
                        help="Number of tokens to generate")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")

    # Logging
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")

    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Create config
    config = SentimentReinforceConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        max_new_tokens=args.max_new_tokens,
        device=args.device
    )

    print("Configuration:")
    print(config)
    print()

    # Load model
    print(f"Loading model: {config.model_name}")
    model = Transformer.from_pretrained(config.model_name)
    print(f"Model loaded: {model.get_num_params() / 1e6:.2f}M parameters")
    print()

    # Create dataloader
    print(f"Loading dataset: {config.dataset_name}")
    dataloader = create_sentiment_dataloader(
        dataset_name=config.dataset_name,
        split=config.dataset_split,
        num_samples=config.num_samples,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    print(f"Dataset loaded: {len(dataloader.dataset)} samples")
    print()

    # Create trainer
    trainer = ReinforceTrainer(
        model=model,
        dataloader=dataloader,
        config=config,
        device=config.device,
        use_wandb=args.wandb
    )

    # Train
    metrics = trainer.train()

    # Print final results
    print("\nFinal Results:")
    for epoch_metrics in metrics:
        print(f"Epoch {epoch_metrics['epoch']+1}: "
              f"Loss={epoch_metrics['avg_loss']:.4f}, "
              f"Reward={epoch_metrics['avg_reward']:.4f}")


if __name__ == "__main__":
    main()
