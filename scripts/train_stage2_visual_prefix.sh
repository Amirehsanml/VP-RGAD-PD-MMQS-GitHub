#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
SEED=42 \
MODEL_NAME_OR_DIR=artifacts/checkpoints/BART_backbone_stage1 \
DATA_PATH=${DATA_PATH:-data/multimodal_final_updated.csv} \
SOURCE_COLUMN=Question TARGET_COLUMN=Question_summ \
VISUAL_EMB_PATH=${VISUAL_EMB_PATH:-data/vgg_image_vector.pt} \
MAX_EPOCHS=5 BATCH_SIZE=4 GRAD_ACCUM_STEPS=8 \
BASE_LR=1e-6 PREFIX_LR=1e-4 WEIGHT_DECAY=0 GRAD_CLIP_NORM=0.5 \
VISUAL_PREFIX_K=8 VISUAL_DROPOUT=0.1 USE_VISUAL_NORMALIZATION=1 \
LABEL_SMOOTHING=0.0 USE_SLIDING_UL=0 REPETITION_LOSS_WEIGHT=0.0 \
USE_RGAD_PREF_TRAIN=0 USE_RGAD_RERANK=0 \
GEN_BEAMS=4 GEN_NO_REPEAT=3 GEN_REP_PEN=1.2 \
OUT_DIR=artifacts/checkpoints/G2_visual_prefix \
OUT_CSV=results/predictions/G2_test_no_rgad.csv \
python src/vp_rgad_pd/run_visual_prefix_bart_rgad.py
