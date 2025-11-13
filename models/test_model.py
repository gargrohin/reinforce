import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from models import Transformer, GPTConfig
import gc

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("gpt2")
    hf = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    gpt = Transformer.from_pretrained("gpt2").to(device).eval()

    # Debug: Check parameter counts
    print(f"\nHF params: {sum(p.numel() for p in hf.parameters()):,}")
    print(f"Custom params: {sum(p.numel() for p in gpt.parameters()):,}")

    print("\nSample weight comparison:")
    print(f"HF first embedding weight: {hf.transformer.wte.weight[0, :5]}")
    print(f"Custom first embedding weight: {gpt.transformer.wte.weight[0, :5]}")
    

    # test
    input = tok("I am gonna build.", return_tensors="pt").to(device)
    with torch.no_grad():
        out_hf = hf(**input).logits
        out_gpt , _ = gpt(input["input_ids"].to(device))
    
    print(f"\nOutput shapes - HF: {out_hf.shape}, Custom: {out_gpt.shape}")
    print(f"Max diff: {(out_hf - out_gpt).abs().max().item()}")
    print(f"Mean diff: {(out_hf - out_gpt).abs().mean().item()}")

    print("\n=== Debug Info ===")
    # print(f"\nHF using SDPA: {hf.config.attn_implementation if hasattr(hf.config, 'attn_implementation') else 'default'}")
    # print(f"HF model training mode: {hf.training}")
    # print(f"Custom model training mode: {gpt.training}")

    # print("\n=== Weight Verification ===")
    # emb_diff = (hf.transformer.wte.weight - gpt.transformer.wte.weight).abs().max()
    # print(f"Embedding weight max diff: {emb_diff}")

    # # Check first layer attention weights
    # hf_attn = hf.transformer.h[0].attn.c_attn.weight
    # gpt_attn = gpt.transformer.h[0].attn.c_attn.weight
    # attn_diff = (hf_attn.t() - gpt_attn).abs().max()  # Note: .t() because of Conv1D
    # print(f"First layer attention weight max diff: {attn_diff}")

    # # Check output head
    # head_diff = (hf.lm_head.weight - gpt.lm_head.weight).abs().max()
    # print(f"LM head weight max diff: {head_diff}")

    # # Check what attention mask HF is using
    # print(f"\nHF attention mask in forward: {input.get('attention_mask')}")

    # # Check layer norm epsilon
    # print(f"\nHF LayerNorm eps: {hf.transformer.ln_f.eps}")
    # print(f"Custom LayerNorm eps: {gpt.transformer.ln_f.eps}")

    # # Check if Flash Attention is being used
    # print(f"\nUsing Flash Attention: {gpt.transformer.h[0].attn.flash}")

    # # Check dropout values
    # print(f"HF config dropout: {hf.config.resid_pdrop}")
    # print(f"Custom config dropout: {gpt.config.dropout}")

      # After the forward pass, before printing diff:
    print("\n=== Intermediate Activations ===")

    # Get embeddings
    with torch.no_grad():
        hf_emb = hf.transformer.wte(input["input_ids"]) + hf.transformer.wpe(torch.arange(5, device=device))
        gpt_emb = gpt.transformer.wte(input["input_ids"]) + gpt.transformer.wpe(torch.arange(5, device=device))
        print(f"After embeddings max diff: {(hf_emb - gpt_emb).abs().max()}")

        # First layer output
        hf_block0_out = hf.transformer.h[0](hf_emb)[0]
        gpt_block0_out = gpt.transformer.h[0](gpt_emb)
        print(f"After block 0 max diff: {(hf_block0_out - gpt_block0_out).abs().max()}")
    
    print("\n==== Block 0 breakdown====")
    with torch.no_grad():
        # After layer norm 1
        hf_ln1 = hf.transformer.h[0].ln_1(hf_emb)
        gpt_ln1 = gpt.transformer.h[0].ln_1(gpt_emb)
        print(f"After ln_1: {(hf_ln1 - gpt_ln1).abs().max()}")

        # After attention (before adding residual)
        hf_attn_out = hf.transformer.h[0].attn(hf_ln1)[0]
        gpt_attn_out = gpt.transformer.h[0].attn(gpt_ln1)
        print(f"After attention: {(hf_attn_out - gpt_attn_out).abs().max()}")

        # After first residual
        hf_res1 = hf_emb + hf_attn_out
        gpt_res1 = gpt_emb + gpt_attn_out
        print(f"After residual 1: {(hf_res1 - gpt_res1).abs().max()}")

        # After layer norm 2
        hf_ln2 = hf.transformer.h[0].ln_2(hf_res1)
        gpt_ln2 = gpt.transformer.h[0].ln_2(gpt_res1)
        print(f"After ln_2: {(hf_ln2 - gpt_ln2).abs().max()}")

        # After MLP
        hf_mlp_out = hf.transformer.h[0].mlp(hf_ln2)
        gpt_mlp_out = gpt.transformer.h[0].mlp(gpt_ln2)
        print(f"After MLP: {(hf_mlp_out - gpt_mlp_out).abs().max()}")
    
    torch.cuda.empty_cache()
    gc.collect()




main()