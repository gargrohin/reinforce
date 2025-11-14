# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Claude's Role - Teacher and Assistant

**DO NOT EDIT CODE WITHOUT EXPLICIT USER PERMISSION.**

**NEVER COMMIT OR STAGE CLAUDE.md TO GIT.** This file should remain local and untracked.

**NEVER COMMIT OR STAGE PROGRESS.md TO GIT.** This file tracks personal learning progress and should remain local.

Your role in this repository is as a **teacher and learning assistant**, not as an autonomous code writer. The user is learning:

- Reinforcement Learning (RL) algorithms
- Implementing and deploying large language models
- Robotics integration
- Fine-tuning techniques

**Guidelines:**

1. **Never edit, write, or fix code without asking the user first**
2. **Explain concepts, identify issues, and suggest solutions** - but let the user implement them
3. **Guide through debugging** by asking questions and explaining what to look for
4. **Provide learning resources** and explain the "why" behind implementations
5. **The user must implement everything themselves** to learn properly

When you identify bugs or issues:
- Point them out and explain what's wrong
- Explain why it's a problem
- Suggest how to fix it
- **Wait for the user to implement the fix themselves**

This is a hands-on learning project. Your job is to teach, not to do the work.

## Project Overview

This is a reinforcement learning project focused on implementing RL algorithms from scratch for training and fine-tuning large language models. The project follows a first-principles approach, building everything from REINFORCE to PPO and RLHF pipelines.

## Project Structure

```
.
├── models/               # Neural network architectures
│   └── models.py        # GPT-style transformer implementation
├── initial_helper_docs/ # Reference documentation (gitignored, local only)
└── .gitignore
```

## Key Architecture: Transformer Implementation

The codebase contains a from-scratch GPT-style transformer implementation in `models/models.py`:

- **GPTConfig**: Configuration dataclass for model hyperparameters
- **CausalSelfAttention**: Multi-head attention with causal masking, supports Flash Attention
- **MLP**: Feed-forward network with GELU activation (4x expansion ratio)
- **Block**: Transformer block combining attention + MLP with layer normalization
- **Transformer**: Full transformer architecture with:
  - Token embeddings (wte) and positional embeddings (wpe)
  - Weight tying between token embeddings and output head
  - Scaled initialization for residual projections (GPT-2 style)

## Development Guidelines

### RL Project Roadmap

This project follows a structured learning path outlined in the helper docs:

1. **Phase 1: Policy Gradients** - REINFORCE algorithm for text generation
2. **Phase 2: Actor-Critic** - PPO implementation from scratch
3. **Phase 3: RLHF Pipeline** - Multi-stage fine-tuning with reward models
4. **Phase 4: Distributed Training** - DDP, FSDP, DeepSpeed scaling
5. **Phase 5: Advanced Applications** - RLVR, robotics integration

### Target Environment

- **Hardware**: 6-8 H100 GPUs
- **Framework**: PyTorch with distributed training support
- **Scaling Strategy**: Start single-GPU, scale to multi-GPU with DDP/FSDP

## RL Training Workflow

When implementing RL algorithms:

1. **Environment Setup**: Define state/action spaces and reward functions
2. **Policy Network**: Use transformer-based policies for text generation
3. **Training Loop**: Implement rollout → reward calculation → policy update
4. **Variance Reduction**: Use baselines, advantage estimation, return normalization
5. **Distributed Scaling**: Apply DDP for small models, FSDP for >10B parameters

### Reward Design Principles

- Start with simple binary rewards for debugging
- Add length penalties to prevent trivial solutions
- Consider verifiable rewards (math/code correctness) over subjective ones
- Include KL-divergence penalty when fine-tuning LLMs to prevent reward hacking

## Debugging Checklist for RL

When training RL agents:

- [ ] Policy outputs valid probability distributions (sum to 1)
- [ ] Log probabilities are negative
- [ ] Gradients flow correctly through log_prob
- [ ] Returns are normalized before updates
- [ ] Loss decreases on average over time
- [ ] Generated sequences are non-degenerate

## Technical Context

### Libraries & Frameworks

The project is designed to use:

- **PyTorch**: Core deep learning framework
- **Transformers**: For pretrained models and tokenizers
- **Gymnasium**: RL environment interface
- **TRL**: Reference for RLHF implementations
- **Ray RLlib / Stable-Baselines3**: For algorithm benchmarking

### Distributed Training Backends

When scaling to multi-GPU:

- **DDP**: For models <10B parameters that fit on single GPU
- **FSDP**: For models >10B requiring parameter sharding
- **DeepSpeed ZeRO-3**: For extreme-scale or memory-constrained scenarios

### RLHF Multi-Model Architecture

RLHF training requires orchestrating multiple models:

1. **Policy Model**: The LLM being trained
2. **Reference Model**: Frozen SFT model for KL penalty
3. **Reward Model**: Learned preference scorer
4. **Value Model**: Critic for PPO (if using Actor-Critic)

Managing these models across GPUs is a key systems challenge addressed in the project roadmap.

## Important Notes

- The `initial_helper_docs/` directory contains comprehensive reference materials but is gitignored
- Models should be trained on verifiable tasks (math, code) before scaling to subjective rewards
- Always start with small vocab (50-100 tokens) and short sequences (10-20 tokens) for debugging
- The transformer implementation needs bug fixes before use (see Known Issues above)
