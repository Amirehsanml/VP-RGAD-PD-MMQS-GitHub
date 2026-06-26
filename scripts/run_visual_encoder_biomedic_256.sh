#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="/home/user2/ehsan/MMQS_project/From-Sights-to-Insights-Towards-Summarization-of-Multimodal-Clinical-Documents"

STAGE1_DIR="$PROJECT_ROOT/audits/bart_baseline_model"
DATA_PATH="$PROJECT_ROOT/MMQS/Dataset/multimodal_final_updated.csv"

RUN_SCRIPT="$ROOT/code/run_visual_prefix_bart_dual_stream.py"

BASE_OUT="$ROOT/experiments/visual_encoder_ablation_img256"

# embedding همیشه از پوشه ثابت — مستقل از BASE_OUT
EMB_BASE="$ROOT/experiments/visual_encoder_ablation"

mkdir -p "$BASE_OUT" "$ROOT/logs"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_MODULE_LOADING=LAZY

echo "[INFO] GPU CHECK"
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY


# -----------------------------
# CONFIG
# -----------------------------
K_LIST=(8)

PREFIX_LR_STAGE2="2e-4"
PREFIX_LR_STAGE3="1e-4"
VISUAL_DROPOUT="0.05"

MAX_SOURCE_LEN=360
MAX_TARGET_LEN=50

BATCH_SIZE=4
GRAD_ACCUM_STEPS=8

RAW_IMG_SIZE=256


# -----------------------------
# SAFE EMB PATH RESOLVER
# -----------------------------
get_emb_path () {
  local enc="$1"
  case "$enc" in
    vgg)
      echo "$EMB_BASE/vgg/embeddings/vgg_embeddings.pt"
      ;;
    vit)
      echo "$EMB_BASE/vit/embeddings/vit_embeddings.pt"
      ;;
    biomedclip)
      echo "$EMB_BASE/biomedclip/embeddings/biomedclip_embeddings.pt"
      ;;
    *)
      echo "[ERROR] unknown encoder: $enc" >&2
      exit 1
      ;;
  esac
}


# -----------------------------
run_stage2 () {
  local enc="$1"
  local k="$2"

  local emb_path
  emb_path=$(get_emb_path "$enc")

  local run_root="$BASE_OUT/$enc/k${k}"
  local out_dir="$run_root/stage2"
  local out_csv="$run_root/outputs/G2_${enc}_k${k}.csv"
  local log="$run_root/stage2.log"

  mkdir -p "$out_dir" "$run_root/outputs"

  echo
  echo "=================================================="
  echo "[STAGE 2] encoder=$enc | K=$k | RAW_IMG_SIZE=$RAW_IMG_SIZE"
  echo "=================================================="

  CUDA_VISIBLE_DEVICES=0 \
  SEED=42 \
  MODEL_NAME_OR_DIR="$STAGE1_DIR" \
  DATA_PATH="$DATA_PATH" \
  SOURCE_COLUMN=Question TARGET_COLUMN=Question_summ \
  VISUAL_EMB_PATH="$emb_path" \
  RAW_IMG_SIZE=$RAW_IMG_SIZE \
  MAX_EPOCHS=20 \
  BATCH_SIZE=$BATCH_SIZE \
  GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS \
  MAX_SOURCE_LEN=$MAX_SOURCE_LEN \
  MAX_TARGET_LEN=$MAX_TARGET_LEN \
  BASE_LR=1e-6 \
  PREFIX_LR=$PREFIX_LR_STAGE2 \
  WEIGHT_DECAY=0 \
  GRAD_CLIP_NORM=0.5 \
  VISUAL_PREFIX_K=$k \
  VISUAL_DROPOUT=$VISUAL_DROPOUT \
  USE_VISUAL_NORMALIZATION=1 \
  LABEL_SMOOTHING=0 \
  USE_RGAD_PREF_TRAIN=0 \
  USE_RGAD_RERANK=0 \
  GEN_BEAMS=4 \
  VISUAL_ABLATION_MODE=real \
  OUT_DIR="$out_dir" \
  OUT_CSV="$out_csv" \
  python "$RUN_SCRIPT" 2>&1 | tee "$log"

  python - <<'PY'
import torch; torch.cuda.empty_cache()
PY
}


# -----------------------------
run_stage3 () {
  local enc="$1"
  local k="$2"

  local emb_path
  emb_path=$(get_emb_path "$enc")

  local run_root="$BASE_OUT/$enc/k${k}"
  local stage2_dir="$run_root/stage2"
  local out_dir="$run_root/stage3"
  local out_csv="$run_root/outputs/G3_${enc}_k${k}.csv"
  local log="$run_root/stage3.log"

  mkdir -p "$out_dir" "$run_root/outputs"

  if [[ ! -d "$stage2_dir" ]]; then
    echo "[SKIP] missing stage2: $stage2_dir"
    return
  fi

  echo
  echo "=================================================="
  echo "[STAGE 3] encoder=$enc | K=$k | RAW_IMG_SIZE=$RAW_IMG_SIZE"
  echo "=================================================="

  CUDA_VISIBLE_DEVICES=0 \
  SEED=42 \
  MODEL_NAME_OR_DIR="$STAGE1_DIR" \
  LOAD_VISUAL_PREFIX_DIR="$stage2_dir" \
  DATA_PATH="$DATA_PATH" \
  SOURCE_COLUMN=Question TARGET_COLUMN=Question_summ \
  VISUAL_EMB_PATH="$emb_path" \
  RAW_IMG_SIZE=$RAW_IMG_SIZE \
  MAX_EPOCHS=20 \
  BATCH_SIZE=$BATCH_SIZE \
  GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS \
  MAX_SOURCE_LEN=$MAX_SOURCE_LEN \
  MAX_TARGET_LEN=$MAX_TARGET_LEN \
  BASE_LR=8e-7 \
  PREFIX_LR=$PREFIX_LR_STAGE3 \
  WEIGHT_DECAY=0 \
  GRAD_CLIP_NORM=0.5 \
  VISUAL_PREFIX_K=$k \
  VISUAL_DROPOUT=$VISUAL_DROPOUT \
  USE_VISUAL_NORMALIZATION=1 \
  LABEL_SMOOTHING=0.05 \
  \
  USE_SLIDING_UL=1 \
  SLIDING_WINDOW_SIZE=5 \
  REPETITION_LOSS_WEIGHT=0.1 \
  \
  USE_RGAD_PREF_TRAIN=1 \
  RGAD_PREF_WEIGHT=0.05 \
  RGAD_PREF_EVERY_N_STEPS=16 \
  RGAD_PREF_NUM_CANDIDATES=4 \
  RGAD_PREF_NUM_BEAMS=8 \
  RGAD_PREF_MIN_NEW_TOKENS=8 \
  RGAD_UNIGRAM_REPEAT_WEIGHT=0.3 \
  RGAD_BIGRAM_REPEAT_WEIGHT=0.5 \
  RGAD_LENGTH_WEIGHT=0.3 \
  RGAD_SOURCE_OVERLAP_WEIGHT=0.2 \
  RGAD_RANK_WEIGHT=0.0 \
  RGAD_TARGET_LEN=50 \
  \
  USE_RGAD_RERANK=0 \
  GEN_BEAMS=4 \
  GEN_NO_REPEAT=2 \
  GEN_REP_PEN=1.4 \
  VISUAL_ABLATION_MODE=real \
  OUT_DIR="$out_dir" \
  OUT_CSV="$out_csv" \
  python "$RUN_SCRIPT" 2>&1 | tee "$log"

  python - <<'PY'
import torch; torch.cuda.empty_cache()
PY
}


# -----------------------------
# MAIN
# -----------------------------
main () {
  local enc="biomedclip"

  local emb_path
  emb_path=$(get_emb_path "$enc")
  if [[ ! -f "$emb_path" ]]; then
    echo "[ERROR] missing embedding: $emb_path"
    exit 1
  fi

  for k in "${K_LIST[@]}"; do
    echo
    echo "######################################################"
    echo "# ENCODER=$enc  K=$k"
    echo "######################################################"
    run_stage2 "$enc" "$k"
    run_stage3 "$enc" "$k"
  done

  echo
  echo "ALL RUNS COMPLETED"
  echo "OUTPUT: $BASE_OUT"
}

main "$@"