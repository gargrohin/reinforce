import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from models import Transformer, GPTConfig

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("gpt2")
    hf = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    gpt = Transformer.from_pretrained("gpt2").to(device).eval()

    # test
    input = tok("I am gonna build.", return_tensors="pt").to(device)
    with torch.no_grad():
        out_hf = hf(**input).logits
        out_gpt , _ = gpt(input["input_ids"].to(device))
    
    print("max diff: ", (out_hf - out_gpt).abs().max().item())

main()