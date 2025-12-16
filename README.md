# reinforce

RL algorithms implemented from scratch for LLM fine-tuning.

## Experiment: Sentiment-Controlled Generation

Fine-tune GPT-2 to generate text with target sentiment using REINFORCE.

**Task:** Given a sentence fragment and target sentiment (positive/negative), generate a completion that matches the target.

**Reward:** Binary (+1/-1) from DistilBERT sentiment classifier on the full generated text.

**Key components:**
- `reinforce/models/models.py` - GPT-2 wrapper with `generate_with_log_probs()` for policy gradient computation
- `reinforce/training/reinforce_trainer.py` - REINFORCE training loop with KL penalty
- `reinforce/training/reward.py` - Sentiment reward function using DistilBERT
- `reinforce/training/baseline.py` - EMA baseline for variance reduction

**Reward hacking mitigation:**
- KL divergence penalty against frozen reference model prevents mode collapse
- Without KL penalty, model degenerates to repeating single tokens ("ClearClearClear")

**Run training:**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_sentiment_reinforce.py --use_wandb
```

**Test model:**
```bash
python scripts/test_model.py
```

## Next: PPO

Actor-Critic implementation with clipped surrogate objective and GAE.
