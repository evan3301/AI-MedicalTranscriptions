import os, json, random
import numpy as np, pandas as pd, torch
from dataclasses import dataclass
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import torch.optim as optim
from tqdm import tqdm

SEED = 13
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

@dataclass
class Config:
    model_name: str = "distilbert-base-uncased"
    max_len: int = 256
    lr: float = 2e-5
    batch_size: int = 8
    epochs: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    outdir: str = "models/cls_distilbert"
    train_csv: str = "data/train.csv"
    val_csv: str = "data/val.csv"
    test_csv: str = "data/test.csv"

class NoteClsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, label2id):
        self.texts = df["text"].astype(str).tolist()
        self.labels = [label2id[y] for y in df["specialty"].tolist()]
        self.tok = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

def load_splits(cfg: Config):
    tr = pd.read_csv(cfg.train_csv)
    va = pd.read_csv(cfg.val_csv)
    te = pd.read_csv(cfg.test_csv)
    labels = sorted(pd.concat([tr["specialty"], va["specialty"], te["specialty"]]).unique())
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}
    return tr, va, te, label2id, id2label

def evaluate(model, loader, device, id2label):
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            p = out.logits.argmax(-1).detach().cpu().numpy().tolist()
            g = batch["labels"].detach().cpu().numpy().tolist()
            preds += p; gold += g

    # Always evaluate over the full label set (even if some classes
    # don't appear in the current split), so sizes stay consistent.
    labels_list = list(range(len(id2label)))
    target_names = [id2label[i] for i in labels_list]

    report = classification_report(
        gold, preds,
        labels=labels_list,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(gold, preds, labels=labels_list).tolist()
    return report, cm

def main():
    cfg = Config()
    os.makedirs(cfg.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va, te, label2id, id2label = load_splits(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label2id),
        id2label=id2label, label2id=label2id
    ).to(device)

    train_ds = NoteClsDataset(tr, tokenizer, cfg.max_len, label2id)
    val_ds   = NoteClsDataset(va, tokenizer, cfg.max_len, label2id)
    test_ds  = NoteClsDataset(te, tokenizer, cfg.max_len, label2id)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(grouped, lr=cfg.lr)
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps) if total_steps > 0 else 0
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps) if total_steps > 0 else None

    best_f1 = -1.0
    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler: scheduler.step()
            optimizer.zero_grad()
            pbar.set_postfix(loss=float(loss.item()))
        # Validation
        val_report, val_cm = evaluate(model, val_loader, device, id2label)
        macro_f1 = val_report["macro avg"]["f1-score"]
        print(f"Val macro-F1: {macro_f1:.4f}")
        with open(os.path.join(cfg.outdir, f"val_report_epoch{epoch+1}.json"), "w") as f:
            json.dump({"report": val_report, "confusion_matrix": val_cm}, f, indent=2)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            model.save_pretrained(cfg.outdir)
            tokenizer.save_pretrained(cfg.outdir)

    # Final test
    model = AutoModelForSequenceClassification.from_pretrained(cfg.outdir).to(device)
    test_report, test_cm = evaluate(model, test_loader, device, id2label)
    with open(os.path.join(cfg.outdir, "test_report.json"), "w") as f:
        json.dump({"report": test_report, "confusion_matrix": test_cm}, f, indent=2)
    print("Saved best to:", cfg.outdir)

if __name__ == "__main__":
    main()
