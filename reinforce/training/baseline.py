"""
Baseline for variance reduction in REINFORCE.
"""

import torch


class RewardBaseline:
    """
    Moving average baseline for REINFORCE.
    Tracks running mean of rewards to reduce variance.
    """

    def __init__(self, decay=0.99):
        """
        Args:
            decay: Exponential moving average decay factor
                   (0.99 = 99% old value, 1% new value)
        """
        self.decay = decay
        self.value = None

    def update(self, rewards):
        """
        Update baseline with new batch of rewards.

        Args:
            rewards: torch.Tensor of shape (batch_size,)
        """
        batch_mean = rewards.mean().item()
        if self.value is None:
            self.value = batch_mean
        else:
            self.value = self.value*self.decay + batch_mean*(1-self.decay)
    
    def get(self):
        """Get current baseline value."""
        return self.value

    def reset(self):
        """Reset baseline to initial state."""
        self.value = None
