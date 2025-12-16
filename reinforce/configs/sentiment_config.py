"""
Configuration for sentiment-controlled REINFORCE training.
"""

from dataclasses import dataclass


@dataclass
class SentimentReinforceConfig:
    """Configuration for REINFORCE training on sentiment task."""

    # Model
    model_name: str = "gpt2"  # gpt2, gpt2-medium, gpt2-large, gpt2-xl

    # Dataset
    dataset_name: str = "imdb"
    dataset_split: str = "train"
    num_samples: int = None  # None = use all data
    batch_size: int = 16
    num_workers: int = 4

    # Training
    num_epochs: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_new_tokens: int = 10  # Number of tokens to generate per prompt
    temperature: float = 1.0  # Sampling temperature
    kl_coef: float = 0.2  # β coefficient for KL penalty

    # REINFORCE
    baseline_decay: float = 0.99  # Exponential moving average decay for baseline
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"

    # Logging & Checkpointing
    log_interval: int = 100  # Log every N batches
    save_interval: int = 1   # Save checkpoint every N epochs
    checkpoint_dir: str = "./checkpoints/sentiment_reinforce"

    # Device
    device: str = "cuda"  # cuda or cpu


# Default config
default_config = SentimentReinforceConfig()
