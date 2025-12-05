"""
Dataset and DataLoader for sentiment-controlled text generation with REINFORCE.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
import re
import random


def extract_sentences(text, tokenizer, min_tokens=15, max_tokens=100):
    """
    Extract clean sentences from review text.

    Args:
        text: Review text (may contain HTML tags)
        tokenizer: GPT2Tokenizer instance
        min_tokens: Minimum sentence length in tokens
        max_tokens: Maximum sentence length in tokens

    Returns:
        List of cleaned sentences
    """
    # Remove HTML tags
    text = re.sub(r'<br\s*/?>', ' ', text)
    # Split on sentence boundaries
    sentences = re.split(r'[.!?]+', text)
    
    clean_sentences = []
    for s in sentences:
        s = s.strip()
        tokens = tokenizer.encode(s)
        # Keep sentences with 15-100 tokens (good range for prompts)
        if min_tokens <= len(tokens) <= max_tokens:
            clean_sentences.append(s)
    
    return clean_sentences


def process_batch(examples, tokenizer, min_tokens=15, max_tokens=100, fragment_tokens=10):
    """
    Process a batch of reviews and extract sentence fragments.

    Args:
        examples: dict with 'text' key containing list of reviews
        tokenizer: GPT2Tokenizer instance

    Returns:
        dict with 'sentence_fragment' and 'full_sentence' lists
    """
    all_fragments = []
    all_full_sentences = []
    
    # examples['text'] is a list of reviews in the batch
    for review_text in examples['text']:
        sentences = extract_sentences(review_text, tokenizer, min_tokens, max_tokens)
        
        for sentence in sentences:
            tokens = tokenizer.encode(sentence)
            if len(tokens) >= min_tokens:
                fragment_tok = tokens[:fragment_tokens]
                fragment_text = tokenizer.decode(fragment_tok)
                
                all_fragments.append(fragment_text)
                all_full_sentences.append(sentence)
    
    return {
        'sentence_fragment': all_fragments,
        'full_sentence': all_full_sentences,
    }



class SentimentRLDataset(Dataset):
    """
    Dataset for REINFORCE training on sentiment-controlled generation.

    Each sample returns:
        - prompt_text: "{sentiment} review : {fragment}"
        - target_sentiment: "positive" or "negative"
        - sentence_fragment: original fragment without prefix
        - full_sentence: complete sentence for reference
    """

    def __init__(self, prompts_data, tokenizer):
        """
        Args:
            prompts_data: List of dicts with 'sentence_fragment' and 'full_sentence'
            tokenizer: GPT2Tokenizer instance
        """
        self.prompts_data = prompts_data
        self.tokenizer = tokenizer
        self.sentiments = ["positive", "negative"]

    def __len__(self):
        return len(self.prompts_data)

    def __getitem__(self, idx):
        """
        Return a single sample with randomly sampled target sentiment.
        """
        data = self.prompts_data[idx]
        
        # Randomly sample target sentiment for this prompt
        target_sentiment = random.choice(self.sentiments)
        
        # Create prompt: "positive review : I rented this movie from"
        prompt = f"{target_sentiment} review : {data['sentence_fragment']}"
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors='pt')
        
        return {
            'prompt_text': prompt,
            'prompt_tokens': prompt_tokens.squeeze(0),
            'target_sentiment': target_sentiment,
            'sentence_fragment': data['sentence_fragment'],
            'full_sentence': data['full_sentence'],  # For debugging
        }


def collate_fn(batch, tokenizer):
    """
    Collate function to pad prompts to same length within batch.

    Args:
        batch: List of dicts from SentimentRLDataset
        tokenizer: GPT2Tokenizer instance

    Returns:
        Dict with batched and padded tensors
    """
    prompt_texts = [item['prompt_text'] for item in batch]
    target_sentiments = [item['target_sentiment'] for item in batch]
    sentence_fragments = [item['sentence_fragment'] for item in batch]
    full_sentences = [item['full_sentence'] for item in batch]

    # Tokenize with padding
    tokenized = tokenizer(
        prompt_texts,
        padding=True,
        return_tensors='pt',
        truncation=True,
        max_length=512
    )

    return {
        'prompt_text': prompt_texts,
        'prompt_tokens': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'target_sentiment': target_sentiments,
        'sentence_fragment': sentence_fragments,
        'full_sentence': full_sentences,
    }


def create_sentiment_dataloader(
    dataset_name="imdb",
    split="train",
    num_samples=None,
    batch_size=16,
    num_workers=4,
    shuffle=True
):
    """
    Factory function to create sentiment RL dataloader.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to use
        num_samples: Number of samples to use (None = all)
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data

    Returns:
        DataLoader instance
    """
    # Load dataset
    ds = load_dataset(dataset_name, split=split)

    # Limit samples if specified
    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Process dataset - wrap process_batch to pass tokenizer
    def process_fn(examples):
        return process_batch(examples, tokenizer)

    processed_ds = ds.map(
        process_fn,
        batched=True,
        batch_size=1000,
        remove_columns=ds.column_names,
        num_proc=num_workers,
        desc="Extracting sentence fragments"
    )

    prompts_data = processed_ds.to_list()

    # Create dataset
    dataset = SentimentRLDataset(prompts_data, tokenizer)

    # Create collate function with tokenizer
    from functools import partial
    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_with_tokenizer,
        num_workers=0  # Set to 0 to avoid multiprocessing issues with tokenizer
    )

    return dataloader
    
