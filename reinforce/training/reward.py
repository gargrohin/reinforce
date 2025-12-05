"""
Reward computation for sentiment-controlled text generation.
"""

import torch
from transformers import pipeline


class SentimentRewardFunction:
    """
    Reward function based on sentiment classification.

    Compares generated text sentiment to target sentiment.
    """

    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english", device=0):
        """
        Args:
            model_name: HuggingFace sentiment classifier model
            device: Device to run classifier on (0 for cuda:0, -1 for CPU)
        """
        self.classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device
        )

    def compute_rewards(self, sentence_fragments, generated_texts, target_sentiments):
        """
        Compute rewards for generated completions (batched).

        Args:
            sentence_fragments: List of original sentence fragments (without sentiment prefix)
            generated_texts: List of generated completion texts
            target_sentiments: List of target sentiments ("positive" or "negative")

        Returns:
            rewards: torch.Tensor of shape (batch_size,) with reward values
        """
        rewards = []

        full_text = [f + " " + g for f, g in zip(sentence_fragments, generated_texts)]

        results = self.classifier(full_text)

        predicted_sentiments = [result['label'].lower() for result in results]
        target_sentiments = [t.lower() for t in target_sentiments]

        rewards = [1.0 if pred == target else -1.0 for pred, target in zip(predicted_sentiments, target_sentiments)]

        return torch.tensor(rewards, dtype=torch.float32)

    def __call__(self, sentence_fragments, generated_texts, target_sentiments):
        """Alias for compute_rewards."""
        return self.compute_rewards(sentence_fragments, generated_texts, target_sentiments)
