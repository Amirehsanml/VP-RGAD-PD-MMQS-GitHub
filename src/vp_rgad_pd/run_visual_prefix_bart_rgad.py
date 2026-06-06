import os
import re
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BartTokenizerFast, BartForConditionalGeneration, get_linear_schedule_with_warmup
from rouge_score import rouge_scorer
from tqdm import tqdm


# -------------------------
# Config
# -------------------------
SEED = int(os.environ.get("SEED", "42"))
DATA_PATH = os.environ.get("DATA_PATH", "MMQS/Dataset/multimodal_final_updated.csv")
VISUAL_EMB_PATH = os.environ.get("VISUAL_EMB_PATH", "MMQS/Dataset/vgg_image_vector.pt")

SOURCE_COLUMN = os.environ.get("SOURCE_COLUMN", "Question")
TARGET_COLUMN = os.environ.get("TARGET_COLUMN", "Question_summ")

MODEL_NAME_OR_DIR = os.environ.get("MODEL_NAME_OR_DIR", "audits/bart_baseline_model")
OUT_DIR = Path(os.environ.get("OUT_DIR", "audits/visual_prefix_bart_model"))
LOAD_VISUAL_PREFIX_DIR = os.environ.get("LOAD_VISUAL_PREFIX_DIR", "")

MAX_SOURCE_LEN = int(os.environ.get("MAX_SOURCE_LEN", "360"))
MAX_TARGET_LEN = int(os.environ.get("MAX_TARGET_LEN", "96"))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM_STEPS", "8"))
EPOCHS = int(os.environ.get("MAX_EPOCHS", "5"))

BASE_LR = float(os.environ.get("BASE_LR", "1e-6"))
PREFIX_LR = float(os.environ.get("PREFIX_LR", "1e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.06"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP_NORM", "0.5"))

VISUAL_PREFIX_K = int(os.environ.get("VISUAL_PREFIX_K", "8"))
VISUAL_DROPOUT = float(os.environ.get("VISUAL_DROPOUT", "0.1"))
VISUAL_CLAMP_VALUE = float(os.environ.get("VISUAL_CLAMP_VALUE", "10.0"))
USE_VISUAL_NORMALIZATION = os.environ.get("USE_VISUAL_NORMALIZATION", "1") == "1"
VISUAL_ABLATION_MODE = os.environ.get("VISUAL_ABLATION_MODE", "real").lower().strip()

LABEL_SMOOTHING = float(os.environ.get("LABEL_SMOOTHING", "0.0"))

USE_SLIDING_UL = os.environ.get("USE_SLIDING_UL", "0") == "1"
SLIDING_WINDOW_SIZE = int(os.environ.get("SLIDING_WINDOW_SIZE", "10"))
REPETITION_LOSS_WEIGHT = float(os.environ.get("REPETITION_LOSS_WEIGHT", "0.0"))

USE_RGAD_PREF_TRAIN = os.environ.get("USE_RGAD_PREF_TRAIN", "0") == "1"
RGAD_PREF_WEIGHT = float(os.environ.get("RGAD_PREF_WEIGHT", "0.05"))
RGAD_PREF_EVERY_N_STEPS = int(os.environ.get("RGAD_PREF_EVERY_N_STEPS", "16"))
RGAD_PREF_NUM_CANDIDATES = int(os.environ.get("RGAD_PREF_NUM_CANDIDATES", "4"))
RGAD_PREF_NUM_BEAMS = int(os.environ.get("RGAD_PREF_NUM_BEAMS", "8"))
RGAD_PREF_MIN_NEW_TOKENS = int(os.environ.get("RGAD_PREF_MIN_NEW_TOKENS", "8"))

USE_RGAD_RERANK = os.environ.get("USE_RGAD_RERANK", "0") == "1"
RGAD_NUM_CANDIDATES = int(os.environ.get("RGAD_NUM_CANDIDATES", "4"))
GEN_BEAMS = int(os.environ.get("GEN_BEAMS", "4"))
GEN_NO_REPEAT = int(os.environ.get("GEN_NO_REPEAT", "3"))
GEN_REP_PEN = float(os.environ.get("GEN_REP_PEN", "1.2"))
RGAD_MIN_NEW_TOKENS = int(os.environ.get("RGAD_MIN_NEW_TOKENS", "8"))

RGAD_UNIGRAM_REPEAT_WEIGHT = float(os.environ.get("RGAD_UNIGRAM_REPEAT_WEIGHT", "1.0"))
RGAD_BIGRAM_REPEAT_WEIGHT = float(os.environ.get("RGAD_BIGRAM_REPEAT_WEIGHT", "2.0"))
RGAD_LENGTH_WEIGHT = float(os.environ.get("RGAD_LENGTH_WEIGHT", "0.15"))
RGAD_SOURCE_OVERLAP_WEIGHT = float(os.environ.get("RGAD_SOURCE_OVERLAP_WEIGHT", "0.05"))
RGAD_RANK_WEIGHT = float(os.environ.get("RGAD_RANK_WEIGHT", "0.0"))
RGAD_TARGET_LEN = float(os.environ.get("RGAD_TARGET_LEN", "24"))

EVAL_ONLY = os.environ.get("EVAL_ONLY", "0") == "1"
SAVE_PREFIX = os.environ.get("SAVE_PREFIX", "1") == "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# -------------------------
# Text utilities
# -------------------------
def toks(x):
    return re.findall(r"[A-Za-z]+|\d+", str(x).lower())


def rep_stats(texts):
    uni, bi = [], []
    for text in texts:
        t = toks(text)
        uni.append(sum(1 for i in range(1, len(t)) if t[i] == t[i - 1]))
        bi.append(sum(1 for i in range(2, len(t)) if t[i - 2:i] == t[i:i + 2]))
    return {
        "has_unigram_repeat_%": sum(x > 0 for x in uni) / len(uni) * 100,
        "avg_unigram_repeat": sum(uni) / len(uni),
        "has_bigram_repeat_%": sum(x > 0 for x in bi) / len(bi) * 100,
        "avg_bigram_repeat": sum(bi) / len(bi),
    }


def rgad_repetition_rates(text):
    t = toks(text)
    if len(t) <= 1:
        return 0.0, 0.0, len(t)
    uni = sum(1 for i in range(1, len(t)) if t[i] == t[i - 1])
    bi = sum(1 for i in range(2, len(t)) if t[i - 2:i] == t[i:i + 2])
    return uni / max(1, len(t) - 1), bi / max(1, len(t) - 2), len(t)


def rgad_source_overlap(source_text, candidate_text):
    src = set(toks(source_text))
    cand = set(toks(candidate_text))
    if not src or not cand:
        return 0.0
    return len(src & cand) / max(1, len(cand))


def rgad_score(source_text, candidate_text, rank_index):
    uni_rate, bi_rate, length = rgad_repetition_rates(candidate_text)
    len_pen = abs(length - RGAD_TARGET_LEN) / max(1.0, RGAD_TARGET_LEN)
    overlap = rgad_source_overlap(source_text, candidate_text)

    score = 0.0
    score += RGAD_SOURCE_OVERLAP_WEIGHT * overlap
    score -= RGAD_UNIGRAM_REPEAT_WEIGHT * uni_rate
    score -= RGAD_BIGRAM_REPEAT_WEIGHT * bi_rate
    score -= RGAD_LENGTH_WEIGHT * len_pen
    score -= RGAD_RANK_WEIGHT * float(rank_index)
    return score


# -------------------------
# Data
# -------------------------
def sanitize_visual(x):
    x = x.float()
    x = torch.nan_to_num(x, nan=0.0, posinf=VISUAL_CLAMP_VALUE, neginf=-VISUAL_CLAMP_VALUE)
    x = torch.clamp(x, -VISUAL_CLAMP_VALUE, VISUAL_CLAMP_VALUE)
    if USE_VISUAL_NORMALIZATION:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std
    return x


class MMQSDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.src = frame[SOURCE_COLUMN].fillna("").astype(str).tolist()
        self.tgt = frame[TARGET_COLUMN].fillna("").astype(str).tolist()
        self.visual = list(frame["_visual_embedding"].values)
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
        labels = y["input_ids"].squeeze(0)
        labels_for_loss = labels.clone()
        labels_for_loss[labels_for_loss == self.tokenizer.pad_token_id] = -100

        visual = torch.tensor(self.visual[idx], dtype=torch.float32)

        return {
            "input_ids": x["input_ids"].squeeze(0),
            "attention_mask": x["attention_mask"].squeeze(0),
            "labels": labels_for_loss,
            "labels_decode": labels,
            "visual": visual,
            "source_text": self.src[idx],
        }


def collate(batch):
    out = {}
    for key in ["input_ids", "attention_mask", "labels", "labels_decode", "visual"]:
        out[key] = torch.stack([b[key] for b in batch])
    out["source_text"] = [b["source_text"] for b in batch]
    return out


# -------------------------
# Visual Prefix BART
# -------------------------
class VisualPrefixAdapter(nn.Module):
    def __init__(self, visual_dim, d_model, k, dropout):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, k * d_model),
        )

    def forward(self, visual):
        x = self.net(visual)
        return x.view(visual.size(0), self.k, self.d_model)


class VisualPrefixBart(nn.Module):
    def __init__(self, model_name_or_dir, visual_dim=768):
        super().__init__()
        self.base = BartForConditionalGeneration.from_pretrained(model_name_or_dir)
        d_model = self.base.config.d_model
        self.prefix = VisualPrefixAdapter(
            visual_dim=visual_dim,
            d_model=d_model,
            k=VISUAL_PREFIX_K,
            dropout=VISUAL_DROPOUT,
        )

    def build_inputs(self, input_ids, attention_mask, visual):
        visual = sanitize_visual(visual)
        embed_scale = getattr(self.base.model.encoder, "embed_scale", 1.0)
        text_embeds = self.base.model.encoder.embed_tokens(input_ids) * embed_scale
        prefix_embeds = self.prefix(visual)

        inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)

        prefix_mask = torch.ones(
            attention_mask.size(0),
            VISUAL_PREFIX_K,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return inputs_embeds, extended_mask

    def forward(self, input_ids, attention_mask, visual, labels=None):
        inputs_embeds, extended_mask = self.build_inputs(input_ids, attention_mask, visual)
        return self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=labels,
        )

    def generate(self, input_ids, attention_mask, visual, **gen_kwargs):
        inputs_embeds, extended_mask = self.build_inputs(input_ids, attention_mask, visual)
        return self.base.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            **gen_kwargs,
        )


# -------------------------
# Losses
# -------------------------
def label_smoothed_ce(logits, labels):
    if LABEL_SMOOTHING <= 0:
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        label_smoothing=LABEL_SMOOTHING,
    )


def sliding_window_ul(logits, labels):
    if not USE_SLIDING_UL or REPETITION_LOSS_WEIGHT <= 0 or SLIDING_WINDOW_SIZE <= 0:
        return logits.new_tensor(0.0)

    B, T, V = logits.shape
    valid = labels != -100
    safe = labels.clone()
    safe[~valid] = 0

    probs = torch.softmax(logits, dim=-1)
    losses = []

    for offset in range(1, min(SLIDING_WINDOW_SIZE, T - 1) + 1):
        cur = safe[:, offset:]
        prev = safe[:, :-offset]
        mask = valid[:, offset:] & valid[:, :-offset] & (cur != prev)

        if not mask.any():
            continue

        step_probs = probs[:, offset:, :]
        p_prev = step_probs.gather(dim=-1, index=prev.unsqueeze(-1)).squeeze(-1)
        p_prev = p_prev.clamp(1e-6, 1 - 1e-6)
        losses.append(-torch.log(1 - p_prev)[mask])

    if not losses:
        return logits.new_tensor(0.0)

    return torch.cat(losses).mean()


def sequence_logprob(model, tokenizer, input_ids, attention_mask, visual, target_texts):
    enc = tokenizer(
        target_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TARGET_LEN,
    )
    pref_labels = enc["input_ids"].to(device)
    pref_labels[pref_labels == tokenizer.pad_token_id] = -100

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        visual=visual,
        labels=pref_labels,
    )

    logits = out.logits
    valid = pref_labels != -100
    safe = pref_labels.masked_fill(~valid, 0)

    logp = F.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(dim=-1, index=safe.unsqueeze(-1)).squeeze(-1)
    seq_logp = (tok_logp * valid.float()).sum(dim=-1) / valid.float().sum(dim=-1).clamp_min(1.0)
    return seq_logp


def rgad_pref_loss(model, tokenizer, input_ids, attention_mask, visual, sources):
    if not USE_RGAD_PREF_TRAIN or RGAD_PREF_WEIGHT <= 0:
        return visual.new_tensor(0.0)

    was_training = model.training
    model.eval()

    num_candidates = max(2, RGAD_PREF_NUM_CANDIDATES)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual=visual,
            num_beams=max(RGAD_PREF_NUM_BEAMS, num_candidates),
            num_return_sequences=num_candidates,
            no_repeat_ngram_size=GEN_NO_REPEAT,
            repetition_penalty=GEN_REP_PEN,
            min_new_tokens=RGAD_PREF_MIN_NEW_TOKENS,
            early_stopping=True,
            max_length=MAX_TARGET_LEN,
        )

        candidates = tokenizer.batch_decode(generated, skip_special_tokens=True)

    if was_training:
        model.train()

    pos_texts, neg_texts, keep_idx = [], [], []

    for i, src in enumerate(sources):
        group = candidates[i * num_candidates:(i + 1) * num_candidates]
        scored = [(rgad_score(src, c, r), r, c.strip()) for r, c in enumerate(group)]
        scored = [x for x in scored if x[2]]
        if len(scored) < 2:
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        pos = scored[0][2]
        neg = scored[-1][2]

        if pos == neg:
            continue

        pos_texts.append(pos)
        neg_texts.append(neg)
        keep_idx.append(i)

    if not keep_idx:
        return visual.new_tensor(0.0)

    idx = torch.tensor(keep_idx, dtype=torch.long, device=device)
    sub_input = input_ids.index_select(0, idx)
    sub_mask = attention_mask.index_select(0, idx)
    sub_visual = visual.index_select(0, idx)

    pos_logp = sequence_logprob(model, tokenizer, sub_input, sub_mask, sub_visual, pos_texts)
    neg_logp = sequence_logprob(model, tokenizer, sub_input, sub_mask, sub_visual, neg_texts)

    return -F.logsigmoid(pos_logp - neg_logp).mean()


# -------------------------
# Eval
# -------------------------
def evaluate(model, tokenizer, dl, name, out_csv):
    model.eval()
    refs, preds = [], []

    with torch.no_grad():
        for batch in tqdm(dl, desc=f"eval {name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            visual = batch["visual"].to(device)

            if USE_RGAD_RERANK:
                n = max(2, RGAD_NUM_CANDIDATES)
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    visual=visual,
                    num_beams=max(GEN_BEAMS, n),
                    num_return_sequences=n,
                    no_repeat_ngram_size=GEN_NO_REPEAT,
                    repetition_penalty=GEN_REP_PEN,
                    min_new_tokens=RGAD_MIN_NEW_TOKENS,
                    early_stopping=True,
                    max_length=MAX_TARGET_LEN,
                )
                candidates = tokenizer.batch_decode(generated, skip_special_tokens=True)

                for i, src in enumerate(batch["source_text"]):
                    group = candidates[i * n:(i + 1) * n]
                    scored = [(rgad_score(src, c, r), r, c) for r, c in enumerate(group)]
                    scored.sort(key=lambda x: x[0], reverse=True)
                    preds.append(scored[0][2])
            else:
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    visual=visual,
                    num_beams=GEN_BEAMS,
                    no_repeat_ngram_size=GEN_NO_REPEAT,
                    repetition_penalty=GEN_REP_PEN,
                    early_stopping=True,
                    max_length=MAX_TARGET_LEN,
                )
                preds.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

            refs.extend(tokenizer.batch_decode(batch["labels_decode"], skip_special_tokens=True))

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for r, y in zip(refs, preds):
        s = scorer.score(r, y)
        for k in scores:
            scores[k].append(s[k].fmeasure * 100)

    rouge = {k: sum(v) / len(v) for k, v in scores.items()}
    rep = rep_stats(preds)

    print(name, "ROUGE:", {k: round(v, 4) for k, v in rouge.items()})
    print(name, "REP:", {k: round(v, 4) for k, v in rep.items()})

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"actual": refs, "predicted": preds}).to_csv(out_csv, index=False)
    print("saved:", out_csv)

    return rouge


def save_model(model, tokenizer, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.base.save_pretrained(out_dir / "base")
    tokenizer.save_pretrained(out_dir)
    torch.save(model.prefix.state_dict(), out_dir / "visual_prefix.pt")
    with open(out_dir / "visual_prefix_config.json", "w") as f:
        json.dump({
            "visual_prefix_k": VISUAL_PREFIX_K,
            "visual_dim": 768,
            "max_source_len": MAX_SOURCE_LEN,
            "max_target_len": MAX_TARGET_LEN,
        }, f, indent=2)


def load_visual_prefix_if_needed(model):
    if not LOAD_VISUAL_PREFIX_DIR:
        return

    path = Path(LOAD_VISUAL_PREFIX_DIR) / "visual_prefix.pt"
    if not path.exists():
        raise FileNotFoundError(path)

    model.prefix.load_state_dict(torch.load(path, map_location="cpu"))
    print("Loaded visual prefix:", path)


# -------------------------
# Main
# -------------------------
print("device:", device)
print("MODEL_NAME_OR_DIR:", MODEL_NAME_OR_DIR)
print("LOAD_VISUAL_PREFIX_DIR:", LOAD_VISUAL_PREFIX_DIR)
print("USE_RGAD_PREF_TRAIN:", USE_RGAD_PREF_TRAIN)
print("USE_RGAD_RERANK:", USE_RGAD_RERANK)
print("SOURCE/TARGET:", SOURCE_COLUMN, TARGET_COLUMN)

df = pd.read_csv(DATA_PATH)
visual = torch.load(VISUAL_EMB_PATH, map_location="cpu").detach().cpu().float()
print("df rows:", len(df), "visual shape:", tuple(visual.shape), "requires_grad:", visual.requires_grad)
print("VISUAL_ABLATION_MODE:", VISUAL_ABLATION_MODE)

if len(df) != visual.size(0):
    raise RuntimeError("visual embedding length does not match dataframe length")

if VISUAL_ABLATION_MODE == "zero":
    visual = torch.zeros_like(visual)
    print("Applied visual ablation: zero embeddings")
elif VISUAL_ABLATION_MODE == "shuffle":
    g = torch.Generator()
    g.manual_seed(SEED)
    perm = torch.randperm(visual.size(0), generator=g)
    visual = visual[perm]
    print("Applied visual ablation: shuffled embeddings")
elif VISUAL_ABLATION_MODE == "real":
    print("Using real visual embeddings")
else:
    raise ValueError(f"Unknown VISUAL_ABLATION_MODE: {VISUAL_ABLATION_MODE}")

df = df.copy()
df["_visual_embedding"] = list(visual.numpy())

train_df, temp_df = train_test_split(df, test_size=0.20, random_state=SEED, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.75, random_state=SEED, shuffle=True)
print("split:", len(train_df), len(val_df), len(test_df))

tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME_OR_DIR if not LOAD_VISUAL_PREFIX_DIR else LOAD_VISUAL_PREFIX_DIR)
model_init_dir = MODEL_NAME_OR_DIR

if LOAD_VISUAL_PREFIX_DIR:
    model_init_dir = str(Path(LOAD_VISUAL_PREFIX_DIR) / "base")

model = VisualPrefixBart(model_init_dir).to(device)
load_visual_prefix_if_needed(model)

train_loader = DataLoader(MMQSDataset(train_df, tokenizer), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
val_loader = DataLoader(MMQSDataset(val_df, tokenizer), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
test_loader = DataLoader(MMQSDataset(test_df, tokenizer), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

if EVAL_ONLY:
    evaluate(model, tokenizer, test_loader, "test", os.environ.get("OUT_CSV", "audits/visual_prefix_outputs/test_eval.csv"))
    raise SystemExit(0)

param_groups = [
    {"params": model.base.parameters(), "lr": BASE_LR},
    {"params": model.prefix.parameters(), "lr": PREFIX_LR},
]

optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
total_update_steps = max(1, (len(train_loader) // max(1, GRAD_ACCUM) + 1) * EPOCHS)
warmup = max(1, int(WARMUP_RATIO * total_update_steps))
scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_update_steps)

best_score = -1

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_loader, desc=f"train epoch {epoch}")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        visual = batch["visual"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual=visual,
            labels=labels,
        )

        logits = out.logits
        ce = label_smoothed_ce(logits, labels)
        ul = sliding_window_ul(logits, labels)
        pref = logits.new_tensor(0.0)

        if USE_RGAD_PREF_TRAIN and RGAD_PREF_EVERY_N_STEPS > 0 and step % RGAD_PREF_EVERY_N_STEPS == 0:
            pref = rgad_pref_loss(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual=visual,
                sources=batch["source_text"],
            )

        loss = ce + REPETITION_LOSS_WEIGHT * ul + RGAD_PREF_WEIGHT * pref

        if not torch.isfinite(loss).item():
            print("NON-FINITE LOSS", loss.item())
            raise SystemExit(77)

        (loss / GRAD_ACCUM).backward()
        running += loss.item()

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    print(f"epoch {epoch} train_loss {running / len(train_loader):.6f}")

    val_csv = f"audits/visual_prefix_outputs/val_epoch{epoch}.csv"
    val = evaluate(model, tokenizer, val_loader, "val", val_csv)
    score = val["rouge1"] + val["rouge2"] + val["rougeL"]

    if score > best_score:
        best_score = score
        if SAVE_PREFIX:
            save_model(model, tokenizer, OUT_DIR)
        print("saved best:", OUT_DIR, "score:", best_score)

print("Loading best and final test")
best = VisualPrefixBart(str(OUT_DIR / "base")).to(device)
best.prefix.load_state_dict(torch.load(OUT_DIR / "visual_prefix.pt", map_location="cpu"))
evaluate(best, tokenizer, test_loader, "test", os.environ.get("OUT_CSV", "audits/visual_prefix_outputs/test_best.csv"))
