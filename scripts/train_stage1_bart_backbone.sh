#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
SEED=42 \
MODEL_NAME=facebook/bart-base \
DATA_PATH=${DATA_PATH:-data/multimodal_final_updated.csv} \
SOURCE_COLUMN=Question TARGET_COLUMN=Question_summ \
MAX_EPOCHS=10 BATCH_SIZE=8 GRAD_ACCUM_STEPS=4 LR=5e-5 \
MAX_SOURCE_LEN=360 MAX_TARGET_LEN=96 \
GEN_BEAMS=4 GEN_NO_REPEAT=3 GEN_REP_PEN=1.2 \
OUT_DIR=artifacts/checkpoints/BART_backbone_stage1 \
python src/vp_rgad_pd/run_bart_backbone.py
