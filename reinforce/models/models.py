import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from transformers import GPT2LMHeadModel
import inspect

# class LayerNorm(nn.Module):

#     def __init__(self, features, bias=True):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(features))
#         self.bias = nn.Parameter(torch.zeros(features)) if bias else None
    
#     def forward(self, x):
#         return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)
# can just set bias = False to pytorch layer norm

@dataclass
class GPTConfig:
    block_size: int  = 1024
    vocab_size: int = 50304 # 50257 for GPT-2 uncased, 50280 for GPT-2 cased, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key query value projections for all heads, in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection layer
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        if not self.flash:
            # causal mask (is this already in flash attention?)
            self.register_buffer("mask", 
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size))
        



    def forward(self, x, attention_mask=None):
        B, T, C = x.size() # batch size, seq length, embed dim (n_embd)

        # query key values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # reshape q, k, v for head attention computation
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, head_dim)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, head_dim)

        # causal self attention: self attend: (B, nh, T, head_dim) x (B, nh, head_dim, T) -> (B, nh, T, T)

        if self.flash:
            # use flash attention cuda kernels
            if attention_mask is not None:
                attn_mask_flash = attention_mask[:, None, None, :].bool()
            else:
                attn_mask_flash = None
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_flash, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # actual stuff
            attn = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
            attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            if attention_mask is not None:
                padding_mask = attention_mask[:, None, None, :]
                attn = attn.masked_fill(padding_mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)

            y = attn @ v # (B, nh, T, T) x (B, nh, T, head_dim) -> (B, nh, T, head_dim)

        y = y.transpose(1,2).contiguous().view(B, T, C)

        #output projection
        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 4 is a parameter transformer gods chose
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    



class Transformer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        #init weights
        self.apply(self._init_weights)

        # scaled init to residual projections:
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*config.n_layer))
        
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def get_num_params(self, non_embeddings=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embeddings:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    
    def generate_with_log_probs(self, idx, max_new_tokens, temperature = 1.0, attention_mask = None):
        """
        Gen tokens and return log probs
        """

        generated_tokens = []
        log_probs = []

        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx, attention_mask=attention_mask)
            logits = logits[:, -1, :]  / temperature

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples = 1)

            log_probs_dist = F.log_softmax(logits, dim=-1)
            log_prob_gathered = log_probs_dist.gather(dim=-1, index=idx_next)

            generated_tokens.append(idx_next)
            log_probs.append(log_prob_gathered)

            idx = torch.cat([idx, idx_next], dim=1)

            # update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,   # Match dtype explicitly
                        device=attention_mask.device
                    )
                ], dim=1)

        return torch.cat(generated_tokens, dim=1), torch.cat(log_probs, dim=1)

    def forward(self, idx, targets=None, attention_mask=None):
        b, t = idx.size()
        device = idx.device
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        #forward
        tok_embd = self.transformer.wte(idx) # b, t, n_embd
        pos_embd = self.transformer.wpe(pos) # t, n_embd

        x = self.transformer.drop(tok_embd + pos_embd)
        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask)
        x = self.transformer.ln_f(x)

        if targets is not None: 
            # calc loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1 , logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            # inference time
            logits = self.lm_head(x)[:, [-1], :] # list [-1] preserves time dimensionnnnn
            loss = None
        
        return logits, loss
    
    def crop_block_size(self, block_size):
        # model surgery bwahaha
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'mask'):
                block.attn.mask = block.attn.mask[:, :, :block_size, :block_size]
    
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):

        override_args = override_args or {}
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        
        assert all(k == 'dropout' for k in override_args)

        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints

        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = Transformer(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.mask')] # discard the mask / buffer, not a param

        # huggingface

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy params
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        
        for k in sd_keys_hf:
            v = sd_hf[k]
            if any(k.endswith(w) for w in transposed):
                v = v.t()

            assert v.shape == sd[k].shape, k
            
            with torch.no_grad():
                sd[k].copy_(v)
        
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):

        param_dict = {pn: p for pn, p in self.named_parameters()}

        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups, to map weight decay to 2D params only.
        # all weight tensors in matmuls + embeddings decay, biases and layernorms dont. 

        decay_params = [p for n, p in param_dict.items() if p.dim() >=2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() <2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def compute_log_probs(self, input_ids, generated_ids):
        """
        Compute log probs for generated ids given input ids as context.

        input_ids: (B, prompt_len + gen_len)
        gen ids: (B, gen_len)

        returns: (B, gen_len) log probs
        """

        logits, _ = self(input_ids, input_ids)
        logits = logits[:, -(generated_ids.shape[1]+1):-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.gather(dim=-1, index=generated_ids.unsqueeze(-1))
        return log_probs.squeeze(-1)


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, top_k=None):
        
        for _ in range(max_new_tokens):
            #crop if too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            #forward
            logits, _ = self(idx_cond)
            # final step
            logits = logits[:, -1, :] / temperature

            # crop for top k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf") # mask
            
            # softmax
            probs = F.softmax(logits, dim=-1)

            # sample and append
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
             

    






        

