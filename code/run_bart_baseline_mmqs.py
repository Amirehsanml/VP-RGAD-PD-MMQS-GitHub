import os
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import BartTokenizerFast, BartForConditionalGeneration, get_linear_schedule_with_warmup
from rouge_score import rouge_scorer
from tqdm import tqdm

SEED = int(os.environ.get("SEED", "42"))
DATA_PATH = os.environ.get("DATA_PATH", "MMQS/Dataset/multimodal_final_updated.csv")
SOURCE_COLUMN = os.environ.get("SOURCE_COLUMN", "Question")
TARGET_COLUMN = os.environ.get("TARGET_COLUMN", "Question_summ")

MAX_SOURCE_LEN = int(os.environ.get("MAX_SOURCE_LEN", "360"))
MAX_TARGET_LEN = int(os.environ.get("MAX_TARGET_LEN", "96"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM_STEPS", "4"))
EPOCHS = int(os.environ.get("MAX_EPOCHS", "10"))
LR = float(os.environ.get("LR", "5e-5"))
MODEL_NAME = os.environ.get("MODEL_NAME", "facebook/bart-base")
OUT_DIR = Path(os.environ.get("OUT_DIR", "audits/bart_baseline_model"))
GEN_BEAMS = int(os.environ.get("GEN_BEAMS", "4"))
GEN_NO_REPEAT = int(os.environ.get("GEN_NO_REPEAT", "3"))
GEN_REP_PEN = float(os.environ.get("GEN_REP_PEN", "1.2"))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("model:", MODEL_NAME)
print("source:", SOURCE_COLUMN, "target:", TARGET_COLUMN)

df = pd.read_csv(DATA_PATH)
train_df, temp_df = train_test_split(df, test_size=0.20, random_state=SEED, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.75, random_state=SEED, shuffle=True)
print("split:", len(train_df), len(val_df), len(test_df))

tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

class SummDS(Dataset):
    def __init__(self, frame):
        self.src = frame[SOURCE_COLUMN].fillna("").astype(str).tolist()
        self.tgt = frame[TARGET_COLUMN].fillna("").astype(str).tolist()

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        x = tokenizer(
            self.src[idx],
            max_length=MAX_SOURCE_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        y = tokenizer(
            self.tgt[idx],
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = y["input_ids"].squeeze(0)
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": x["input_ids"].squeeze(0),
            "attention_mask": x["attention_mask"].squeeze(0),
            "labels": labels,
        }

def loader(frame, shuffle):
    return DataLoader(SummDS(frame), batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = loader(train_df, True)
val_loader = loader(val_df, False)
test_loader = loader(test_df, False)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps = (len(train_loader) // max(1, GRAD_ACCUM) + 1) * EPOCHS
warmup = max(1, int(0.06 * total_steps))
scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)

def evaluate(frame, dl, name, epoch):
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"eval {name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=GEN_BEAMS,
                no_repeat_ngram_size=GEN_NO_REPEAT,
                repetition_penalty=GEN_REP_PEN,
                early_stopping=True,
                max_length=MAX_TARGET_LEN,
            )
            preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))

            lab = labels.numpy()
            lab = np.where(lab != -100, lab, tokenizer.pad_token_id)
            refs.extend(tokenizer.batch_decode(lab, skip_special_tokens=True))

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for r, y in zip(refs, preds):
        s = scorer.score(r, y)
        for k in scores:
            scores[k].append(s[k].fmeasure*100)

    out = {k: sum(v)/len(v) for k,v in scores.items()}
    print(name, "epoch", epoch, {k: round(v,4) for k,v in out.items()})

    Path("audits/bart_outputs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"actual": refs, "predicted": preds}).to_csv(f"audits/bart_outputs/{name}_epoch{epoch}.csv", index=False)
    return out

best = -1
OUT_DIR.mkdir(parents=True, exist_ok=True)

for epoch in range(1, EPOCHS+1):
    model.train()
    total = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_loader, desc=f"train epoch {epoch}")):
        batch = {k:v.to(device) for k,v in batch.items()}
        out = model(**batch)
        loss = out.loss / GRAD_ACCUM
        loss.backward()
        total += loss.item() * GRAD_ACCUM

        if (step+1) % GRAD_ACCUM == 0 or (step+1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    print("epoch", epoch, "train_loss", total/len(train_loader))
    val = evaluate(val_df, val_loader, "val", epoch)
    score = val["rouge1"] + val["rouge2"] + val["rougeL"]

    if score > best:
        best = score
        model.save_pretrained(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)
        print("saved best", OUT_DIR, "score", best)

print("Loading best and final test")
model = BartForConditionalGeneration.from_pretrained(OUT_DIR).to(device)
evaluate(test_df, test_loader, "test", "best")
