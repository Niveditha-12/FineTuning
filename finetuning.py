import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from llama.model import ModelArgs, Llama
from llama.lora import apply_lora  # function to modify Q/V with LoRA


# ---------- tokenizer
import os
from llama.tokenizer import Tokenizer

class WrappedTiktokenTokenizer:
    def __init__(self, model_path: str, max_length: int = 512):
        self.tokenizer = Tokenizer(model_path)
        self.max_length = max_length
        self.pad_id = 0

    def __call__(self, text, max_length=None):
        max_length = max_length or self.max_length
        input_ids = self.tokenizer.encode(text, bos=False, eos=True)
        input_ids = input_ids[:max_length]

        pad_len = max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [self.pad_id] * pad_len

        # 🟢 Force return as tensors (not lists)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }



    def encode(self, text, bos=False, eos=True):
        return self.tokenizer.encode(text, bos=bos, eos=eos)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

checkppoint_dir = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B")
#checkpoint_dir = "checkpoints/llama-3.2-1B"  # <-- update this to match your actual folder
tokenizer_path = os.path.join(checkppoint_dir, "tokenizer.model")

tokenizer = WrappedTiktokenTokenizer(tokenizer_path, max_length=512)


# -------- Dataset (Alpaca) --------
class AlpacaDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_length=512):
        with open(data_path, "r") as f:
            raw_data = json.load(f)

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []

        for example in raw_data:
            instruction = example["instruction"]
            input_text = example.get("input", "")
            output_text = example["output"]

            if input_text.strip():
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

            full_text = prompt + output_text

            tokenized = tokenizer(full_text, max_length=self.max_seq_length)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            labels = input_ids.clone()

            # Prompt masking
            prompt_tokenized = tokenizer(prompt, max_length=self.max_seq_length)
            prompt_len = prompt_tokenized["input_ids"].numel()
            labels[:prompt_len] = -100


            self.examples.append((input_ids, attention_mask, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# -------- Training Loop --------
def train(model, dataloader, accumulation_steps=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
    )
    scaler = torch.cuda.amp.GradScaler()

    step = 0
    loss_log = []

    for epoch in range(1):
        for i, (input_ids, attn_mask, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
                )
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                print(f"Step {step} | Loss: {loss.item() * accumulation_steps:.4f}")
                loss_log.append(loss.item() * accumulation_steps)
                step += 1

    torch.save(model.state_dict(), "finetuned_llama_lora.pth")
    return loss_log


# -------- Main --------
if __name__ == "__main__":
    # ✅ Keep your custom tokenizer
    # tokenizer already defined above using WrappedTiktokenTokenizer

    dataset = AlpacaDataset("data/alpaca_200.json", tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    base_model = Llama(...)  # fill with your model args
    model = apply_lora(base_model, r=16, alpha=32, dropout=0.05)

    for name, module in model.named_modules():
        if "wq" in name or "wv" in name:
            print(f"{name} → {type(module)}")

    loss_log = train(model, dataloader)

    with open("loss_log.txt", "w") as f:
        for val in loss_log:
            f.write(f"{val}\n")

