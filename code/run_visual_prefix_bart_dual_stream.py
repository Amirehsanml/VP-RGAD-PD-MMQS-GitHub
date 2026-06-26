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
from PIL import Image as PILImage
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import (
    BartTokenizerFast,
    BartForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from rouge_score import rouge_scorer
from tqdm import tqdm


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SEED               = int(os.environ.get("SEED", "42"))
DATA_PATH          = os.environ.get("DATA_PATH", "MMQS/Dataset/multimodal_final_updated.csv")
VISUAL_EMB_PATH    = os.environ.get("VISUAL_EMB_PATH", "MMQS/Dataset/vgg_image_vector.pt")
IMAGE_ROOT         = os.environ.get("IMAGE_ROOT", "")          # ریشه تصاویر روی دیسک
IMAGE_PATH_COLUMN  = os.environ.get("IMAGE_PATH_COLUMN", "relative_path")

# اندازه resize تصویر خام قبل از flatten
# dim pixel stream = RAW_IMG_SIZE * RAW_IMG_SIZE * 3
RAW_IMG_SIZE       = int(os.environ.get("RAW_IMG_SIZE", "256"))  # 32×32×3 = 3072
RAW_PIXEL_DIM      = RAW_IMG_SIZE * RAW_IMG_SIZE * 3

SOURCE_COLUMN      = os.environ.get("SOURCE_COLUMN", "Question")
TARGET_COLUMN      = os.environ.get("TARGET_COLUMN", "Question_summ")

MODEL_NAME_OR_DIR  = os.environ.get("MODEL_NAME_OR_DIR", "audits/bart_baseline_model")
OUT_DIR            = Path(os.environ.get("OUT_DIR", "audits/visual_prefix_dual_stream"))
LOAD_VISUAL_PREFIX_DIR = os.environ.get("LOAD_VISUAL_PREFIX_DIR", "")

MAX_SOURCE_LEN     = int(os.environ.get("MAX_SOURCE_LEN", "360"))
MAX_TARGET_LEN     = int(os.environ.get("MAX_TARGET_LEN", "50"))

BATCH_SIZE         = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM         = int(os.environ.get("GRAD_ACCUM_STEPS", "8"))
EPOCHS             = int(os.environ.get("MAX_EPOCHS", "5"))

BASE_LR            = float(os.environ.get("BASE_LR", "1e-6"))
PREFIX_LR          = float(os.environ.get("PREFIX_LR", "1e-4"))
WEIGHT_DECAY       = float(os.environ.get("WEIGHT_DECAY", "0.0"))
WARMUP_RATIO       = float(os.environ.get("WARMUP_RATIO", "0.06"))
GRAD_CLIP          = float(os.environ.get("GRAD_CLIP_NORM", "0.5"))

VISUAL_PREFIX_K    = int(os.environ.get("VISUAL_PREFIX_K", "8"))
VISUAL_DROPOUT     = float(os.environ.get("VISUAL_DROPOUT", "0.1"))
VISUAL_CLAMP_VALUE = float(os.environ.get("VISUAL_CLAMP_VALUE", "10.0"))
USE_VISUAL_NORM    = os.environ.get("USE_VISUAL_NORMALIZATION", "1") == "1"
VISUAL_ABLATION    = os.environ.get("VISUAL_ABLATION_MODE", "real").lower().strip()

# self-attention هر stream
SA_DIM             = int(os.environ.get("SA_DIM", "256"))   # بُعد داخلی SA
SA_HEADS           = int(os.environ.get("SA_HEADS", "8"))   # تعداد head

LABEL_SMOOTHING    = float(os.environ.get("LABEL_SMOOTHING", "0.0"))

USE_SLIDING_UL         = os.environ.get("USE_SLIDING_UL", "0") == "1"
SLIDING_WINDOW_SIZE    = int(os.environ.get("SLIDING_WINDOW_SIZE", "10"))
REPETITION_LOSS_WEIGHT = float(os.environ.get("REPETITION_LOSS_WEIGHT", "0.0"))

USE_RGAD_PREF_TRAIN      = os.environ.get("USE_RGAD_PREF_TRAIN", "0") == "1"
RGAD_PREF_WEIGHT         = float(os.environ.get("RGAD_PREF_WEIGHT", "0.05"))
RGAD_PREF_EVERY_N_STEPS  = int(os.environ.get("RGAD_PREF_EVERY_N_STEPS", "16"))
RGAD_PREF_NUM_CANDIDATES = int(os.environ.get("RGAD_PREF_NUM_CANDIDATES", "4"))
RGAD_PREF_NUM_BEAMS      = int(os.environ.get("RGAD_PREF_NUM_BEAMS", "8"))
RGAD_PREF_MIN_NEW_TOKENS = int(os.environ.get("RGAD_PREF_MIN_NEW_TOKENS", "8"))

USE_RGAD_RERANK   = os.environ.get("USE_RGAD_RERANK", "0") == "1"
RGAD_NUM_CANDIDATES = int(os.environ.get("RGAD_NUM_CANDIDATES", "4"))
GEN_BEAMS         = int(os.environ.get("GEN_BEAMS", "4"))
GEN_NO_REPEAT     = int(os.environ.get("GEN_NO_REPEAT", "3"))
GEN_REP_PEN       = float(os.environ.get("GEN_REP_PEN", "1.2"))
RGAD_MIN_NEW_TOKENS = int(os.environ.get("RGAD_MIN_NEW_TOKENS", "8"))

RGAD_UNIGRAM_REPEAT_WEIGHT  = float(os.environ.get("RGAD_UNIGRAM_REPEAT_WEIGHT", "1.0"))
RGAD_BIGRAM_REPEAT_WEIGHT   = float(os.environ.get("RGAD_BIGRAM_REPEAT_WEIGHT", "2.0"))
RGAD_LENGTH_WEIGHT          = float(os.environ.get("RGAD_LENGTH_WEIGHT", "0.15"))
RGAD_SOURCE_OVERLAP_WEIGHT  = float(os.environ.get("RGAD_SOURCE_OVERLAP_WEIGHT", "0.05"))
RGAD_RANK_WEIGHT            = float(os.environ.get("RGAD_RANK_WEIGHT", "0.0"))
RGAD_TARGET_LEN             = float(os.environ.get("RGAD_TARGET_LEN", "24"))

EVAL_ONLY    = os.environ.get("EVAL_ONLY", "0") == "1"
SAVE_PREFIX  = os.environ.get("SAVE_PREFIX", "1") == "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ─────────────────────────────────────────────
# Stream Self-Attention
# ─────────────────────────────────────────────
class StreamSelfAttention(nn.Module):
    """
    یه وکتور تخت (B, in_dim) می‌گیره.
    - project به SA_DIM
    - n_virtual تا learnable query می‌سازه
    - MultiheadAttention: query=virtual tokens, key=value=feature
    - pool → خروجی (B, SA_DIM)

    هیچ مدل encoder جدیدی لود نمی‌کنه.
    """
    def __init__(self, in_dim: int, sa_dim: int, n_heads: int, dropout: float = 0.1, n_virtual: int = 4):
        super().__init__()
        assert sa_dim % n_heads == 0, "SA_DIM باید بر SA_HEADS بخش‌پذیر باشه"
        self.proj   = nn.Linear(in_dim, sa_dim)
        self.norm   = nn.LayerNorm(sa_dim)
        self.vq     = nn.Parameter(torch.randn(1, n_virtual, sa_dim) * 0.02)
        self.attn   = nn.MultiheadAttention(sa_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2  = nn.LayerNorm(sa_dim)
        self.ff     = nn.Sequential(
            nn.Linear(sa_dim, sa_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sa_dim * 2, sa_dim),
        )
        self.norm3  = nn.LayerNorm(sa_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        h  = self.norm(self.proj(x)).unsqueeze(1)          # (B, 1, sa_dim)
        vq = self.vq.expand(x.size(0), -1, -1)             # (B, n_virtual, sa_dim)
        a, _ = self.attn(vq, h, h)                         # (B, n_virtual, sa_dim)
        a  = self.norm2(a + vq)
        a  = self.norm3(self.ff(a) + a)
        return a.mean(dim=1)                                # (B, sa_dim)


# ─────────────────────────────────────────────
# Dual-Stream Visual Prefix Adapter
# ─────────────────────────────────────────────
class DualStreamAdapter(nn.Module):
    """
    stream A : pre-computed embedding  (B, emb_dim)   → SA → (B, SA_DIM)
    stream B : raw pixel flatten       (B, pixel_dim)  → SA → (B, SA_DIM)
    concat → (B, 2*SA_DIM) → MLP → (B, k, d_model)
    """
    def __init__(self, emb_dim: int, pixel_dim: int, d_model: int, k: int, dropout: float):
        super().__init__()
        self.sa_emb   = StreamSelfAttention(emb_dim,   SA_DIM, SA_HEADS, dropout)
        self.sa_pixel = StreamSelfAttention(pixel_dim, SA_DIM, SA_HEADS, dropout)

        fused = SA_DIM * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, k * d_model),
        )
        self.k = k
        self.d_model = d_model 

    def forward(self, emb: torch.Tensor, pixel: torch.Tensor) -> torch.Tensor:
        h_emb   = self.sa_emb(emb)          # (B, SA_DIM)
        h_pixel = self.sa_pixel(pixel)       # (B, SA_DIM)
        fused   = torch.cat([h_emb, h_pixel], dim=-1)   # (B, 2*SA_DIM)
        out     = self.fusion(fused)         # (B, k*d_model)
        return out.view(emb.size(0), self.k, self.d_model)


# ─────────────────────────────────────────────
# Text utilities  (بدون تغییر)
# ─────────────────────────────────────────────
def toks(x):
    return re.findall(r"[A-Za-z]+|\d+", str(x).lower())

def rep_stats(texts):
    uni, bi = [], []
    for text in texts:
        t = toks(text)
        if len(t) == 0:
            uni.append(0); bi.append(0); continue
        # distinct-n based
        uni_rep = 1 - len(set(t)) / len(t)
        bigrams = list(zip(t, t[1:]))
        bi_rep  = 1 - (len(set(bigrams)) / len(bigrams)) if bigrams else 0
        uni.append(uni_rep)
        bi.append(bi_rep)
    return {
        "avg_unigram_rep_rate": sum(uni)/len(uni)*100,
        "avg_bigram_rep_rate":  sum(bi)/len(bi)*100,
    }
    
def rgad_repetition_rates(text):
    t = toks(text)
    if len(t) <= 1:
        return 0.0, 0.0, len(t)
    uni = sum(1 for i in range(1, len(t)) if t[i] == t[i-1])
    bi  = sum(1 for i in range(2, len(t)) if t[i-2:i] == t[i:i+2])
    return uni / max(1, len(t)-1), bi / max(1, len(t)-2), len(t)

def rgad_source_overlap(src, cand):
    s, c = set(toks(src)), set(toks(cand))
    return len(s & c) / max(1, len(c)) if s and c else 0.0

def rgad_score(src, cand, rank):
    ur, br, length = rgad_repetition_rates(cand)
    lp = abs(length - RGAD_TARGET_LEN) / max(1.0, RGAD_TARGET_LEN)
    ov = rgad_source_overlap(src, cand)
    return (RGAD_SOURCE_OVERLAP_WEIGHT * ov
            - RGAD_UNIGRAM_REPEAT_WEIGHT * ur
            - RGAD_BIGRAM_REPEAT_WEIGHT  * br
            - RGAD_LENGTH_WEIGHT         * lp
            - RGAD_RANK_WEIGHT           * float(rank))


# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────
def sanitize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = torch.nan_to_num(x, nan=0.0, posinf=VISUAL_CLAMP_VALUE, neginf=-VISUAL_CLAMP_VALUE)
    x = torch.clamp(x, -VISUAL_CLAMP_VALUE, VISUAL_CLAMP_VALUE)
    if USE_VISUAL_NORM:
        m = x.mean(dim=-1, keepdim=True)
        s = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = (x - m) / s
    return x

def resolve_path(image_root: str, rel: str) -> Path:
    rel = str(rel).strip().replace("\\", "/").lstrip("./")
    rel = rel.replace("Multimodal_images/Multimodal_images/", "Multimodal_images/")
    return Path(image_root) / rel

def load_raw_pixel(path: Path) -> torch.Tensor:
    """
    تصویر را resize=RAW_IMG_SIZE×RAW_IMG_SIZE می‌کند،
    به float32 [0,1] تبدیل و flatten می‌کند → (RAW_PIXEL_DIM,)
    هیچ مدلی استفاده نمی‌شود.
    """
    try:
        img = PILImage.open(path).convert("RGB").resize(
            (RAW_IMG_SIZE, RAW_IMG_SIZE), PILImage.BILINEAR
        )
        arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)
        return torch.from_numpy(arr.flatten())           # (H*W*3,)
    except Exception:
        return torch.zeros(RAW_PIXEL_DIM)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class MMQSDualDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.src    = frame[SOURCE_COLUMN].fillna("").astype(str).tolist()
        self.tgt    = frame[TARGET_COLUMN].fillna("").astype(str).tolist()
        self.emb    = list(frame["_visual_embedding"].values)
        self.tokenizer = tokenizer

        self.img_paths = []
        if IMAGE_ROOT and IMAGE_PATH_COLUMN in frame.columns:
            self.img_paths = [
                resolve_path(IMAGE_ROOT, p)
                for p in frame[IMAGE_PATH_COLUMN].astype(str).tolist()
            ]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        # tokenize
        x = self.tokenizer(
            self.src[idx], max_length=MAX_SOURCE_LEN,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        y = self.tokenizer(
            self.tgt[idx], max_length=MAX_TARGET_LEN,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        labels = y["input_ids"].squeeze(0)
        labels_loss = labels.clone()
        labels_loss[labels_loss == self.tokenizer.pad_token_id] = -100

        # stream A — pre-computed embedding
        emb = torch.tensor(self.emb[idx], dtype=torch.float32)

        # stream B — raw pixel (resize+flatten, no encoder)
        if self.img_paths:
            pixel = load_raw_pixel(self.img_paths[idx])
        else:
            pixel = torch.zeros(RAW_PIXEL_DIM)

        return {
            "input_ids":      x["input_ids"].squeeze(0),
            "attention_mask": x["attention_mask"].squeeze(0),
            "labels":         labels_loss,
            "labels_decode":  labels,
            "emb":            emb,
            "pixel":          pixel,
            "source_text":    self.src[idx],
        }

def collate(batch):
    out = {}
    for k in ["input_ids", "attention_mask", "labels", "labels_decode", "emb", "pixel"]:
        out[k] = torch.stack([b[k] for b in batch])
    out["source_text"] = [b["source_text"] for b in batch]
    return out


# ─────────────────────────────────────────────
# Dual-Stream Visual Prefix BART
# ─────────────────────────────────────────────
class DualStreamVisualPrefixBart(nn.Module):
    def __init__(self, model_name_or_dir: str, emb_dim: int):
        super().__init__()
        self.base = BartForConditionalGeneration.from_pretrained(model_name_or_dir)
        d_model   = self.base.config.d_model
        self.prefix = DualStreamAdapter(
            emb_dim   = emb_dim,
            pixel_dim = RAW_PIXEL_DIM,
            d_model   = d_model,
            k         = VISUAL_PREFIX_K,
            dropout   = VISUAL_DROPOUT,
        )

    def _build(self, input_ids, attention_mask, emb, pixel):
        emb   = sanitize(emb)
        pixel = sanitize(pixel)

        scale  = getattr(self.base.model.encoder, "embed_scale", 1.0)
        t_emb  = self.base.model.encoder.embed_tokens(input_ids) * scale
        p_emb  = self.prefix(emb, pixel)                        # (B, k, d_model)

        inputs_embeds  = torch.cat([p_emb, t_emb], dim=1)
        prefix_mask    = torch.ones(
            attention_mask.size(0), VISUAL_PREFIX_K,
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        extended_mask  = torch.cat([prefix_mask, attention_mask], dim=1)
        return inputs_embeds, extended_mask

    def forward(self, input_ids, attention_mask, emb, pixel, labels=None):
        ie, em = self._build(input_ids, attention_mask, emb, pixel)
        return self.base(inputs_embeds=ie, attention_mask=em, labels=labels)

    def generate(self, input_ids, attention_mask, emb, pixel, **kw):
        ie, em = self._build(input_ids, attention_mask, emb, pixel)
        return self.base.generate(inputs_embeds=ie, attention_mask=em, **kw)


# ─────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────
def label_smoothed_ce(logits, labels):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1),
        ignore_index=-100,
        label_smoothing=LABEL_SMOOTHING if LABEL_SMOOTHING > 0 else 0.0,
    )

def sliding_window_ul(logits, labels):
    if not USE_SLIDING_UL or REPETITION_LOSS_WEIGHT <= 0:
        return logits.new_tensor(0.0)
    B, T, V = logits.shape
    valid = labels != -100
    safe  = labels.clone(); safe[~valid] = 0
    probs = torch.softmax(logits, dim=-1)
    losses = []
    for off in range(1, min(SLIDING_WINDOW_SIZE, T-1) + 1):
        cur  = safe[:, off:]
        prev = safe[:, :-off]
        mask = valid[:, off:] & valid[:, :-off] & (cur != prev)
        if not mask.any():
            continue
        p = probs[:, off:, :].gather(-1, prev.unsqueeze(-1)).squeeze(-1).clamp(1e-6, 1-1e-6)
        losses.append(-torch.log(1 - p)[mask])
    return torch.cat(losses).mean() if losses else logits.new_tensor(0.0)

def seq_logprob(model, tokenizer, input_ids, attention_mask, emb, pixel, texts):
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_TARGET_LEN)
    lbl = enc["input_ids"].to(device)
    lbl[lbl == tokenizer.pad_token_id] = -100
    out  = model(input_ids=input_ids, attention_mask=attention_mask,
                 emb=emb, pixel=pixel, labels=lbl)
    valid = lbl != -100
    safe  = lbl.masked_fill(~valid, 0)
    logp  = F.log_softmax(out.logits, dim=-1)
    tok_logp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * valid.float()).sum(-1) / valid.float().sum(-1).clamp_min(1.0)

def rgad_pref_loss(model, tokenizer, input_ids, attention_mask, emb, pixel, sources):
    if not USE_RGAD_PREF_TRAIN or RGAD_PREF_WEIGHT <= 0:
        return emb.new_tensor(0.0)
    was_train = model.training; model.eval()
    n = max(2, RGAD_PREF_NUM_CANDIDATES)
    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, emb=emb, pixel=pixel,
            num_beams=max(RGAD_PREF_NUM_BEAMS, n), num_return_sequences=n,
            no_repeat_ngram_size=GEN_NO_REPEAT, repetition_penalty=GEN_REP_PEN,
            min_new_tokens=RGAD_PREF_MIN_NEW_TOKENS, early_stopping=True,
            max_length=MAX_TARGET_LEN,
        )
        cands = tokenizer.batch_decode(gen, skip_special_tokens=True)
    if was_train: model.train()
    pos_t, neg_t, keep = [], [], []
    for i, src in enumerate(sources):
        grp = cands[i*n:(i+1)*n]
        sc  = [(rgad_score(src, c, r), c.strip()) for r, c in enumerate(grp)]
        sc  = [x for x in sc if x[1]]
        if len(sc) < 2: continue
        sc.sort(key=lambda x: x[0], reverse=True)
        if sc[0][1] == sc[-1][1]: continue
        pos_t.append(sc[0][1]); neg_t.append(sc[-1][1]); keep.append(i)
    if not keep: return emb.new_tensor(0.0)
    idx  = torch.tensor(keep, dtype=torch.long, device=device)
    si, sm, se, sp = (t.index_select(0, idx) for t in (input_ids, attention_mask, emb, pixel))
    pos_lp = seq_logprob(model, tokenizer, si, sm, se, sp, pos_t)
    neg_lp = seq_logprob(model, tokenizer, si, sm, se, sp, neg_t)
    return -F.logsigmoid(pos_lp - neg_lp).mean()


# ─────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────
def evaluate(model, tokenizer, dl, name, out_csv):
    model.eval(); refs, preds = [], []
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"eval {name}"):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            emb  = batch["emb"].to(device)
            pix  = batch["pixel"].to(device)
            if USE_RGAD_RERANK:
                n   = max(2, RGAD_NUM_CANDIDATES)
                gen = model.generate(
                    input_ids=ids, attention_mask=mask, emb=emb, pixel=pix,
                    num_beams=max(GEN_BEAMS, n), num_return_sequences=n,
                    no_repeat_ngram_size=GEN_NO_REPEAT, repetition_penalty=GEN_REP_PEN,
                    min_new_tokens=RGAD_MIN_NEW_TOKENS, early_stopping=True,
                    max_length=MAX_TARGET_LEN,
                )
                cands = tokenizer.batch_decode(gen, skip_special_tokens=True)
                for i, src in enumerate(batch["source_text"]):
                    grp = cands[i*n:(i+1)*n]
                    sc  = sorted([(rgad_score(src, c, r), c) for r, c in enumerate(grp)],
                                 key=lambda x: x[0], reverse=True)
                    preds.append(sc[0][1])
            else:
                gen = model.generate(
                    input_ids=ids, attention_mask=mask, emb=emb, pixel=pix,
                    num_beams=GEN_BEAMS, no_repeat_ngram_size=GEN_NO_REPEAT,
                    repetition_penalty=GEN_REP_PEN, early_stopping=True,
                    max_length=MAX_TARGET_LEN,
                )
                preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
            refs.extend(tokenizer.batch_decode(batch["labels_decode"], skip_special_tokens=True))

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for r, y in zip(refs, preds):
        s = scorer.score(r, y)
        for k in scores: scores[k].append(s[k].fmeasure * 100)
    rouge = {k: sum(v)/len(v) for k, v in scores.items()}
    rep   = rep_stats(preds)
    print(name, "ROUGE:", {k: round(v, 4) for k, v in rouge.items()})
    print(name, "REP:",   {k: round(v, 4) for k, v in rep.items()})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"actual": refs, "predicted": preds}).to_csv(out_csv, index=False)
    print("saved:", out_csv)
    return rouge


def save_model(model, tokenizer, out_dir, emb_dim):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model.base.save_pretrained(out_dir / "base")
    tokenizer.save_pretrained(out_dir)
    torch.save(model.prefix.state_dict(), out_dir / "visual_prefix.pt")
    with open(out_dir / "visual_prefix_config.json", "w") as f:
        json.dump({
            "visual_prefix_k": VISUAL_PREFIX_K,
            "emb_dim":         emb_dim,
            "raw_img_size":    RAW_IMG_SIZE,
            "pixel_dim":       RAW_PIXEL_DIM,
            "sa_dim":          SA_DIM,
            "sa_heads":        SA_HEADS,
            "max_source_len":  MAX_SOURCE_LEN,
            "max_target_len":  MAX_TARGET_LEN,
        }, f, indent=2)


def load_prefix_if_needed(model):
    if not LOAD_VISUAL_PREFIX_DIR: return
    p = Path(LOAD_VISUAL_PREFIX_DIR) / "visual_prefix.pt"
    if not p.exists(): raise FileNotFoundError(p)
    model.prefix.load_state_dict(torch.load(p, map_location="cpu"))
    print("Loaded visual prefix:", p)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
print("device:", device)
print("MODEL_NAME_OR_DIR:", MODEL_NAME_OR_DIR)
print("VISUAL_EMB_PATH:", VISUAL_EMB_PATH)
print(f"Stream A: pre-computed embedding")
print(f"Stream B: raw pixel {RAW_IMG_SIZE}×{RAW_IMG_SIZE}×3 = {RAW_PIXEL_DIM}  [no encoder]")
print(f"SA_DIM={SA_DIM}  SA_HEADS={SA_HEADS}  K={VISUAL_PREFIX_K}")

# لود embedding
df     = pd.read_csv(DATA_PATH)
visual = torch.load(VISUAL_EMB_PATH, map_location="cpu").detach().cpu().float()
EMB_DIM = visual.size(1)
print(f"df rows={len(df)}  visual shape={tuple(visual.shape)}")

if len(df) != visual.size(0):
    raise RuntimeError("visual embedding length != dataframe length")

if VISUAL_ABLATION == "zero":
    visual = torch.zeros_like(visual); print("Ablation: zero")
elif VISUAL_ABLATION == "shuffle":
    g = torch.Generator(); g.manual_seed(SEED)
    visual = visual[torch.randperm(visual.size(0), generator=g)]; print("Ablation: shuffle")
elif VISUAL_ABLATION == "real":
    print("Using real embeddings")
else:
    raise ValueError(f"Unknown VISUAL_ABLATION_MODE: {VISUAL_ABLATION}")

df = df.copy()
df["_visual_embedding"] = list(visual.numpy())

# split
train_df, temp_df = train_test_split(df, test_size=0.20, random_state=SEED, shuffle=True)
val_df,  test_df  = train_test_split(temp_df, test_size=0.75, random_state=SEED, shuffle=True)
print(f"split: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

# tokenizer & model
tok_dir = LOAD_VISUAL_PREFIX_DIR if LOAD_VISUAL_PREFIX_DIR else MODEL_NAME_OR_DIR
tokenizer = BartTokenizerFast.from_pretrained(tok_dir)

init_dir = str(Path(LOAD_VISUAL_PREFIX_DIR) / "base") if LOAD_VISUAL_PREFIX_DIR else MODEL_NAME_OR_DIR
model = DualStreamVisualPrefixBart(init_dir, emb_dim=EMB_DIM).to(device)
load_prefix_if_needed(model)

# dataloaders
train_loader = DataLoader(MMQSDualDataset(train_df, tokenizer), batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
val_loader   = DataLoader(MMQSDualDataset(val_df,   tokenizer), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
test_loader  = DataLoader(MMQSDualDataset(test_df,  tokenizer), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

if EVAL_ONLY:
    evaluate(model, tokenizer, test_loader, "test",
             os.environ.get("OUT_CSV", "audits/dual_stream_outputs/test_eval.csv"))
    raise SystemExit(0)

# optimizer
param_groups = [
    {"params": model.base.parameters(),   "lr": BASE_LR},
    {"params": model.prefix.parameters(), "lr": PREFIX_LR},
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
total_steps = max(1, (len(train_loader) // max(1, GRAD_ACCUM) + 1) * EPOCHS)
warmup      = max(1, int(WARMUP_RATIO * total_steps))
scheduler   = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

best_score = -1.0

for epoch in range(1, EPOCHS + 1):
    model.train(); running = 0.0; optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_loader, desc=f"train epoch {epoch}")):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl  = batch["labels"].to(device)
        emb  = batch["emb"].to(device)
        pix  = batch["pixel"].to(device)

        out    = model(input_ids=ids, attention_mask=mask, emb=emb, pixel=pix, labels=lbl)
        ce     = label_smoothed_ce(out.logits, lbl)
        ul     = sliding_window_ul(out.logits, lbl)
        pref   = out.logits.new_tensor(0.0)

        if USE_RGAD_PREF_TRAIN and RGAD_PREF_EVERY_N_STEPS > 0 and step % RGAD_PREF_EVERY_N_STEPS == 0:
            pref = rgad_pref_loss(model, tokenizer, ids, mask, emb, pix, batch["source_text"])

        loss = ce + REPETITION_LOSS_WEIGHT * ul + RGAD_PREF_WEIGHT * pref

        if not torch.isfinite(loss):
            print("NON-FINITE LOSS", loss.item()); raise SystemExit(77)

        (loss / GRAD_ACCUM).backward()
        running += loss.item()

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

    print(f"epoch {epoch}  train_loss={running/len(train_loader):.6f}")

    val = evaluate(model, tokenizer, val_loader, "val",
                   f"audits/dual_stream_outputs/val_epoch{epoch}.csv")
    score = val["rouge1"] + val["rouge2"] + val["rougeL"]

    if score > best_score:
        best_score = score
        if SAVE_PREFIX:
            save_model(model, tokenizer, OUT_DIR, EMB_DIM)
        print(f"saved best → {OUT_DIR}  score={best_score:.4f}")

print("Loading best model for final test ...")
best = DualStreamVisualPrefixBart(str(OUT_DIR / "base"), emb_dim=EMB_DIM).to(device)
best.prefix.load_state_dict(torch.load(OUT_DIR / "visual_prefix.pt", map_location="cpu"))
evaluate(best, tokenizer, test_loader, "test",
         os.environ.get("OUT_CSV", "audits/dual_stream_outputs/test_best.csv"))