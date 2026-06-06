#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate 2>/dev/null || true

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
SEED=42 \
EVAL_ONLY=1 \
MODEL_NAME_OR_DIR=artifacts/checkpoints/BART_backbone_stage1 \
LOAD_VISUAL_PREFIX_DIR=artifacts/checkpoints/G3_VP_RGAD_PD_final \
DATA_PATH=${DATA_PATH:-data/multimodal_final_updated.csv} \
SOURCE_COLUMN=Question TARGET_COLUMN=Question_summ \
VISUAL_EMB_PATH=${VISUAL_EMB_PATH:-data/vgg_image_vector.pt} \
BATCH_SIZE=4 \
USE_RGAD_RERANK=0 \
GEN_BEAMS=4 GEN_NO_REPEAT=3 GEN_REP_PEN=1.2 \
OUT_CSV=results/predictions/G3_test_no_rgad.csv \
python src/vp_rgad_pd/run_visual_prefix_bart_rgad.py
