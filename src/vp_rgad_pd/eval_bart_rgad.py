import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BartTokenizerFast, BartForConditionalGeneration
from rouge_score import rouge_scorer
from tqdm import tqdm

SEED = int(os.environ.get("SEED", "42"))
DATA_PATH = os.environ.get("DATA_PATH", "MMQS/Dataset/multimodal_final_updated.csv")
MODEL_DIR = os.environ.get("MODEL_DIR", "audits/bart_baseline_model")
SOURCE_COLUMN = os.environ.get("SOURCE_COLUMN", "Question")
TARGET_COLUMN = os.environ.get("TARGET_COLUMN", "Question_summ")

MAX_SOURCE_LEN = int(os.environ.get("MAX_SOURCE_LEN", "360"))
MAX_TARGET_LEN = int(os.environ.get("MAX_TARGET_LEN", "96"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))

NUM_BEAMS = int(os.environ.get("GEN_BEAMS", "8"))
NUM_CANDIDATES = int(os.environ.get("RGAD_NUM_CANDIDATES", "4"))
NO_REPEAT = int(os.environ.get("GEN_NO_REPEAT", "3"))
REP_PEN = float(os.environ.get("GEN_REP_PEN", "1.2"))
MIN_NEW_TOKENS = int(os.environ.get("RGAD_MIN_NEW_TOKENS", "8"))

UW = float(os.environ.get("RGAD_UNIGRAM_REPEAT_WEIGHT", "1.0"))
BW = float(os.environ.get("RGAD_BIGRAM_REPEAT_WEIGHT", "2.0"))
LW = float(os.environ.get("RGAD_LENGTH_WEIGHT", "0.15"))
SW = float(os.environ.get("RGAD_SOURCE_OVERLAP_WEIGHT", "0.05"))
RW = float(os.environ.get("RGAD_RANK_WEIGHT", "0.0"))
TARGET_LEN = float(os.environ.get("RGAD_TARGET_LEN", "24"))

OUT_CSV = os.environ.get("OUT_CSV", "audits/bart_outputs/test_rgad.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def toks(x):
    return re.findall(r"[A-Za-z]+|\d+", str(x).lower())

def rep_features(text):
    t = toks(text)
    if len(t) <= 1:
        return 0.0, 0.0, len(t)
    uni = sum(1 for i in range(1, len(t)) if t[i] == t[i-1])
    bi = sum(1 for i in range(2, len(t)) if t[i-2:i] == t[i:i+2])
    return uni / max(1, len(t)-1), bi / max(1, len(t)-2), len(t)

def source_overlap(src, cand):
    src_set = set(toks(src))
    cand_set = set(toks(cand))
    if not src_set or not cand_set:
        return 0.0
    return len(src_set & cand_set) / max(1, len(cand_set))

def score_candidate(src, cand, rank):
    uni_rate, bi_rate, length = rep_features(cand)
    len_pen = abs(length - TARGET_LEN) / max(1.0, TARGET_LEN)
    overlap = source_overlap(src, cand)
    score = 0.0
    score += SW * overlap
    score -= UW * uni_rate
    score -= BW * bi_rate
    score -= LW * len_pen
    score -= RW * rank
    return score

def rep_stats(texts):
    uni, bi = [], []
    for text in texts:
        t = toks(text)
        uni.append(sum(1 for i in range(1, len(t)) if t[i] == t[i-1]))
        bi.append(sum(1 for i in range(2, len(t)) if t[i-2:i] == t[i:i+2]))
    return {
        "has_unigram_repeat_%": sum(x > 0 for x in uni) / len(uni) * 100,
        "avg_unigram_repeat": sum(uni) / len(uni),
        "has_bigram_repeat_%": sum(x > 0 for x in bi) / len(bi) * 100,
        "avg_bigram_repeat": sum(bi) / len(bi),
    }

class SummDS(Dataset):
    def __init__(self, frame, tokenizer):
        self.src = frame[SOURCE_COLUMN].fillna("").astype(str).tolist()
        self.tgt = frame[TARGET_COLUMN].fillna("").astype(str).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        x = self.tokenizer(
            self.src[idx],
            max_length=MAX_SOURCE_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        y = self.tokenizer(
            self.tgt[idx],
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": x["input_ids"].squeeze(0),
            "attention_mask": x["attention_mask"].squeeze(0),
            "labels": y["input_ids"].squeeze(0),
            "source_text": self.src[idx],
        }

def evaluate(split_name, frame, loader, model, tokenizer):
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval {split_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sources = batch["source_text"]

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=max(NUM_BEAMS, NUM_CANDIDATES),
                num_return_sequences=NUM_CANDIDATES,
                no_repeat_ngram_size=NO_REPEAT,
                repetition_penalty=REP_PEN,
                min_new_tokens=MIN_NEW_TOKENS,
                early_stopping=True,
                max_length=MAX_TARGET_LEN,
            )

            candidates = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for i, src in enumerate(sources):
                group = candidates[i*NUM_CANDIDATES:(i+1)*NUM_CANDIDATES]
                scored = [(score_candidate(src, c, r), r, c) for r, c in enumerate(group)]
                scored.sort(key=lambda x: x[0], reverse=True)
                preds.append(scored[0][2])

            refs.extend(tokenizer.batch_decode(batch["labels"], skip_special_tokens=True))

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for r, y in zip(refs, preds):
        s = scorer.score(r, y)
        for k in scores:
            scores[k].append(s[k].fmeasure * 100)

    rouge = {k: sum(v)/len(v) for k, v in scores.items()}
    rep = rep_stats(preds)

    print(split_name, "ROUGE:", {k: round(v,4) for k,v in rouge.items()})
    print(split_name, "REP:", {k: round(v,4) for k,v in rep.items()})

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"actual": refs, "predicted": preds}).to_csv(OUT_CSV, index=False)
    print("saved:", OUT_CSV)

df = pd.read_csv(DATA_PATH)
train_df, temp_df = train_test_split(df, test_size=0.20, random_state=SEED, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.75, random_state=SEED, shuffle=True)
print("split:", len(train_df), len(val_df), len(test_df))

tokenizer = BartTokenizerFast.from_pretrained(MODEL_DIR)
model = BartForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)

test_loader = DataLoader(SummDS(test_df, tokenizer), batch_size=BATCH_SIZE, shuffle=False)
evaluate("test", test_df, test_loader, model, tokenizer)
