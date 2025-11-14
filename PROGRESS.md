# Project Progress

## Current Goal
Implement GPT-2 Small from scratch (Karpathy-style):
- ✅ Load Hugging Face weights to confirm equivalence
- ✅ Use as REINFORCE policy model for simple text sentiment task

## Current Status

### Completed ✅

#### Phase 1: GPT-2 Implementation and Validation (DONE!)
- ✅ **Fixed bugs in models/models.py:**
  - Line 32: `_init_` → `__init__` (CausalSelfAttention constructor)
  - Line 53: `torch.trill` → `torch.tril` (causal mask)
  - Line 112: `config.nn_embd` → `config.n_embd` (Block layer norm)
  - Line 96: `nn.GELU()` → `nn.GELU(approximate='tanh')` (GPT-2 uses tanh approximation)

- ✅ **Implemented `from_pretrained()` method:**
  - Loads HuggingFace GPT-2 weights (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
  - Maps weight names between HF and custom implementation
  - Handles Conv1D → Linear transpose correctly
  - Filters masked buffers (attn.mask, attn.bias)

- ✅ **Created `models/load_model.py` validation script:**
  - Loads both HF and custom GPT-2 models
  - Compares outputs on same input
  - Achieved validation: **max diff = 7.6e-05** (essentially perfect!)

- ✅ **Debugging session learnings:**
  - Systematically tested: weights → embeddings → each layer → each operation
  - Found GELU approximation mismatch through layer-by-layer comparison
  - Small differences (1e-5) compound through 12 layers
  - Final precision: ~1e-4 is floating-point perfect

- ✅ **Set up project structure** with models/ directory
- ✅ **Created CLAUDE.md** with teaching guidelines
- ✅ **Created PROGRESS.md** for tracking

### In Progress
- [ ] Design REINFORCE text environment
- [ ] Implement sentiment reward function

## Next Steps

### Phase 2: Build REINFORCE Environment (CURRENT)
4. **Create text environment wrapper** (`environments/text_env.py`)
   - Define state/action spaces
   - Implement reset() and step()
   - Handle sequence generation

5. **Implement sentiment reward function:**
   - Use pretrained sentiment classifier (e.g., distilbert-base-uncased-finetuned-sst-2-english)
   - Binary or graded rewards
   - Add length penalties to prevent trivial solutions

### Phase 3: REINFORCE Implementation
6. **Create REINFORCE trainer** (`train_reinforce.py`)
   - Rollout: Generate episodes using policy
   - Compute returns (Monte Carlo)
   - Policy gradient update
   - Logging and checkpointing

7. **Training and debugging:**
   - Start with small vocab (50-100 tokens)
   - Short sequences (10-20 tokens)
   - Monitor policy entropy, loss, average rewards

## Key Learnings

### Transformer Architecture
- **Causal masking:** Each token only attends to previous tokens, maintaining autoregressive property
- **Weight tying:** Input embeddings (wte) and output head (lm_head) share weights, operating in same semantic space
- **Residual projection scaling:** `0.02/sqrt(2*n_layer)` prevents activation explosion in deep networks by keeping variance stable

### GPT-2 Conventions
- Flash Attention when available via `torch.nn.functional.scaled_dot_product_attention`
- Learned positional embeddings (not sinusoidal)
- 4x expansion ratio in MLP
- Layer norm before attention and MLP (pre-norm architecture)
- **CRITICAL:** Uses tanh-approximated GELU, not standard GELU
  - Standard GELU uses exact error function (erf)
  - GPT-2 uses `nn.GELU(approximate='tanh')` for speed
  - Difference: ~0.01-0.02 in outputs, compounds through layers

### Weight Loading (HuggingFace → Custom)
- HuggingFace uses `Conv1D` layers (weights are transposed)
- Custom implementation uses `Linear` layers
- Must transpose weights for: `attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`
- Filter out buffers: `attn.mask`, `attn.masked_bias`, `attn.bias`
- Use `state_dict[k].copy_(hf_state_dict[k].t())` for transposed weights

### Debugging Deep Networks
- Test systematically: weights → embeddings → layer-by-layer → operation-by-operation
- Small numerical differences (1e-5) compound through deep networks (12 layers)
- Final validation target: max diff < 1e-3 (achieved 7.6e-05)

## Resources
- `initial_helper_docs/` - Comprehensive RL and LLM training guides
- Karpathy's nanoGPT: Reference implementation
- HuggingFace Transformers: Official GPT-2 weights and tokenizer

## Notes
- Hardware available: 6-8 H100 GPUs
- Start single-GPU, scale later with DDP/FSDP
- Focus on learning fundamentals before optimization
