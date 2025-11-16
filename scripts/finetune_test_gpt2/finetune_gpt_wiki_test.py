from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from reinforce.models import Transformer, GPTConfig
import wandb

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

block_size = 1024
epochs = 5
learning_rate = 1e-5
batch_size = 16
weight_decay = 0.1
USE_WANDB = True  # Set to True to enable W&B logging

def tokenize_function(examples):
    # Tokenize without adding special tokens, filter empty
    return tok(examples["text"], add_special_tokens=False)

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])

    # Create chunks of size block_size + 1, then split into input/target
    chunk_size = block_size + 1
    num_chunks = total_length // chunk_size

    inputs = []
    targets = []
    for i in range(num_chunks):
        start = i * chunk_size
        chunk = concatenated["input_ids"][start:start + chunk_size]
        inputs.append(chunk[:-1])   # First block_size tokens
        targets.append(chunk[1:])   # Last block_size tokens (shifted by 1)

    result = {
        "input_ids": inputs,
        "targets": targets,
    }
    return result

# Filter out empty texts
ds = ds.filter(lambda x: len(x["text"].strip()) > 0)

# Tokenize all texts
tokenized_ds = ds.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_ds = tokenized_ds.remove_columns(["attention_mask"])

# Group into chunks of block_size
ds = tokenized_ds.map(group_texts, batched=True)
ds.set_format(type='torch', columns=['input_ids', 'targets'])

# No need for DataCollator since all sequences are same length (block_size)
dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

print(f"Dataset size: {len(ds)}")
print(f"Number of batches per epoch: {len(dl)}")
print(f"Sample input_ids lengths: {[len(x) for x in ds['input_ids'][:10]]}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer.from_pretrained("gpt2").to(device)

opt = model.configure_optimizers(
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas=(0.9, 0.95),
    device_type=device.type,
)

# Initialize wandb
if USE_WANDB:
    wandb.init(
        project="gpt2-finetune-test",
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "block_size": block_size,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "dataset": "wikitext-2"
        }
    )

model.train()

for epoch in range(epochs):
    losses = []
    bc = 0
    for batch in tqdm(dl):
        input_ids = batch["input_ids"].to(device)
        # For causal LM, targets are the same as inputs
        targets = batch["targets"].to(device)

        # Debug first batch
        # if bc == 0:
        #     print("\n=== FIRST BATCH DEBUG ===")
        #     print(f"input_ids shape: {input_ids.shape}")
        #     print(f"targets shape: {targets.shape}")
        #     print(f"input_ids[0, :20]: {input_ids[0, :20]}")
        #     print(f"targets[0, :20]: {targets[0, :20]}")
        #     print(f"Number of EOS/PAD tokens (50256) in input_ids: {(input_ids == tok.eos_token_id).sum().item()}")
        #     print(f"Min/Max of input_ids: {input_ids.min().item()}/{input_ids.max().item()}")
        #     print(f"Are input_ids and targets equal: {torch.equal(input_ids, targets)}")
        #     print("=========================\n")

        logits, loss = model(input_ids, targets)

        # Check for issues
        # if bc == 0:
        #     print(f"logits shape: {logits.shape}")
        #     print(f"logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f}")
        #     print(f"Initial loss: {loss.item():.6f}")

        loss.backward()

        # Check gradients
        # if bc == 0:
        #     total_norm = 0
        #     for p in model.parameters():
        #         if p.grad is not None:
        #             param_norm = p.grad.data.norm(2)
        #             total_norm += param_norm.item() ** 2
        #     total_norm = total_norm ** 0.5
        #     print(f"Gradient norm before clipping: {total_norm:.6f}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())

        # if bc % 100 == 0:
        #     print(f"\nStep {bc}: loss: {loss.item():.6f}")
        #     print(f"Sample tokens: {input_ids[0, :20]}")

        bc += 1
        if USE_WANDB:
            wandb.log({"loss": loss.item()})

    mean_loss = sum(losses) / len(losses)
    print(f"Epoch {epoch} loss: {mean_loss}")
    print(f"mean loss: {mean_loss:.4f}")
    if USE_WANDB:
        wandb.log({"epoch": epoch, "mean_loss": mean_loss})


model.eval()
prompt = "I am gonna build "
inp = tok(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(inp["input_ids"], max_new_tokens=50, temperature=1.0, top_k=50)
generated_text = tok.decode(out[0], skip_special_tokens=True)
print(generated_text)

# Log generated sample
if USE_WANDB:
    wandb.log({"generated_text": generated_text})

# Finish wandb run
if USE_WANDB:
    wandb.finish()

