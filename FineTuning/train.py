import os
import sys
import json
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

# rendre import local accessible
sys.path.append('../Model')
sys.path.append('../Tokenizer')
from gpt2_model import GPT2Model
from Tokenizer import MYBPE

class ChatDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        h = self.pairs[idx]['human'].strip()
        a = self.pairs[idx]['assistant'].strip()
        # construire prompt: Human: ... \n Bot:
        prefix = f"Human: {h}\nBot:"
        text = prefix + " " + a
        # encoder (MYBPE.encoder utilisé ailleurs)
        ids_prefix = self.tokenizer.encoder(prefix)
        ids_all = self.tokenizer.encoder(text)
        # limiter longueur
        if len(ids_all) > self.max_length:
            ids_all = ids_all[-self.max_length:]
        # position où commence la réponse (assistant)
        assist_start = max(0, len(ids_all) - len(self.tokenizer.encoder(a)))
        return {
            "input_ids": torch.tensor(ids_all, dtype=torch.long),
            "assist_start": assist_start
        }

def collate_fn(batch, pad_id=None):
    input_ids_list = [b["input_ids"] for b in batch]
    assist_starts = [b["assist_start"] for b in batch]
    max_len = max([t.size(0) for t in input_ids_list])
    if pad_id is None:
        pad_id = 0
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, ids in enumerate(input_ids_list):
        L = ids.size(0)
        input_ids[i, :L] = ids
        attention_mask[i, :L] = 1
        # labels: only contribute for assistant tokens -> keep token ids, mask others as -100
        start = assist_starts[i]
        labels[i, start:L] = input_ids[i, start:L]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def load_conversations(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    train = data.get("train", [])
    val = data.get("val", [])
    return train, val

def load_model_and_tokenizer(model_dir, tokenizer_path, device, config_fallback=None):
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    else:
        cfg = config_fallback or {"vocab_size":300,"embed_dim":128,"num_heads":4,"num_layers":2,"max_seq_len":64}

    tokenizer = MYBPE(vocab_size=cfg.get("vocab_size", 300))
    tokenizer.load_tokenizer(tokenizer_path)

    model = GPT2Model(
        vocab_size=cfg.get("vocab_size", 300),
        embed_dim=cfg.get("embed_dim", 128),
        num_heads=cfg.get("num_heads", 4),
        num_layers=cfg.get("num_layers", 2),
        max_seq_len=cfg.get("max_seq_len", 64)
    )

    model_file = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"model.pt introuvable dans {model_dir}")

    try:
        state = torch.load(model_file, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_file, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.train()
    return model, tokenizer, cfg

def train_loop(model, tokenizer, train_dataset, val_dataset, device, epochs=3, batch_size=8, lr=5e-5, out_dir="./my_tiny_chatbot"):
    pad_id = getattr(tokenizer, "eos_id", None) or getattr(tokenizer, "eos_token_id", None) or 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id))

    optimizer = AdamW(model.parameters(), lr=lr)
    loss_f = CrossEntropyLoss(ignore_index=-100)
    for ep in range(1, epochs+1):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train epoch {ep}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # forward
            logits, _ = model(input_ids)  # logits: (B, L, V)
            # shift logits and labels for causal LM: predict token t given tokens <= t-1
            lm_logits = logits[:, :-1, :].contiguous()
            lm_labels = labels[:, 1:].contiguous()
            lm_loss = loss_f(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))

            optimizer.zero_grad()
            lm_loss.backward()
            optimizer.step()

            total_loss += lm_loss.item()
            pbar.set_postfix({'loss': f'{lm_loss.item():.4f}'})

        avg = total_loss / len(train_loader)
        print(f"Epoch {ep} train loss: {avg:.4f}")

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits, _ = model(input_ids)
                lm_logits = logits[:, :-1, :].contiguous()
                lm_labels = labels[:, 1:].contiguous()
                loss = loss_f(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader) if len(val_loader) else 0.0
        print(f"Epoch {ep} val loss: {avg_val:.4f}")
        model.train()

        # sauvegarde intermédiaire
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, f"model_epoch{ep}.pt"))
    # sauvegarde finale
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    print("Entraînement terminé. Modèle sauvegardé dans", out_dir)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune local GPT2Model with conversations.json")
    parser.add_argument("--data", type=str, default="./conversations.json", help="fichier JSON produit (train+val)")
    parser.add_argument("--model-dir", type=str, default="./my_tiny_chatbot", help="répertoire contenant model.pt/config.json")
    parser.add_argument("--tokenizer", type=str, default="../Tokenizer/tokenizer_model.bin", help="chemin tokenizer")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="./my_tiny_chatbot")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_pairs, val_pairs = [], []
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"{args.data} introuvable")
    t, v = load_conversations(args.data)
    train_pairs = t
    val_pairs = v
    # fallback if file has single list named "train"/"val" missing
    if not train_pairs and isinstance(t, list) and len(t)>0:
        train_pairs = t
    if not val_pairs and isinstance(v, list) and len(v)>0:
        val_pairs = v

    model, tokenizer, cfg = load_model_and_tokenizer(args.model_dir, args.tokenizer, device)
    train_ds = ChatDataset(train_pairs, tokenizer, max_length=cfg.get("max_seq_len", 512))
    val_ds = ChatDataset(val_pairs, tokenizer, max_length=cfg.get("max_seq_len", 512))
    train_loop(model, tokenizer, train_ds, val_ds, device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, out_dir=args.out_dir)

if __name__ == "__main__":
    main()