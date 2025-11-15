from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import torch
from tqdm import tqdm
from reinforce.models import Transformer, GPTConfig
import wandb

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

block_size = 1024
epochs = 2
learning_rate = 3e-5
batch_size = 4
weight_decay = 0.1

def chunk_text(examples):
    joined = tok(examples["text"], truncation=True, max_length = block_size)
    return {"input_ids": joined["input_ids"]}

ds = ds.map(chunk_text, batched=True, remove_columns=['text'])
ds.set_format(type='torch', columns=['input_ids'])

data_collector = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=data_collector)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer.from_pretrained("gpt2").to(device)

opt = model.configure_optimizers(
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas=(0.9, 0.95),
    device_type=device.type,
)

# Initialize wandb
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
    for batch in tqdm(dl):
        input_ids = batch["input_ids"].to(device)
        targets = input_ids.clone()
        logits, loss = model(input_ids, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        wandb.log({"loss": loss.item()})

    mean_loss = sum(losses) / len(losses)
    print(f"Epoch {epoch} loss: {mean_loss}")
    print(f"mean loss: {mean_loss:.4f}")
    wandb.log({"epoch": epoch, "mean_loss": mean_loss})


model.eval()
prompt = "I am gonna build."
inp = tok(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(inp["input_ids"], max_new_tokens=50, temperature=1.0, top_k=50)
generated_text = tok.decode(out[0], skip_special_tokens=True)
print(generated_text)

# Log generated sample
wandb.log({"generated_text": generated_text})

# Finish wandb run
wandb.finish()

