#!/usr/bin/env bash
# SIGMark HV-1.0 T2V SEMANTIC-stratified clean eval – GLOBAL 512-bit.
#
# Uses fr_factor=16 so all 16 latent frames share the same 512-bit message
# (watermark_c=8, watermark_f=1, watermark_h=8, watermark_w=8 → 512 bits).
# This enables a fair-capacity comparison with SKCW at 512-bit (r=1).
#
# Mirrors run_sigmark_hv_semantic.sh (same prompts/seeds/dims).
# Run:  setsid nohup ./run_sigmark_hv_semantic_512bit.sh > run_sigmark_hv_semantic_512bit.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")" || exit 1

# Configure local paths with environment variables when needed.
PY=${PYTHON:-python}
MODEL_BASE=${MODEL_BASE:-./pretrained_models}
MODEL_NAME=${MODEL_NAME:-HunyuanVideo}
OUT=${OUT:-./outputs-hv-t2v/sigmark_semantic_512bit}
PROMPT_SET=${PROMPT_SET:-VBench2_aug_18}
W=512; H=512; NF=65; STEPS=50; FPS=15; GUID=6.0
NPP=2          # num_prompts_per_dimension
NVP=2          # num_videos_per_prompt

echo "=== [$(date)] SIGMark GLOBAL 512-bit 1/2: generate (clean, all dims) ==="
echo "Using Python: $PY ($($PY --version))"
$PY main.py --mode gen \
  --model_base_path "$MODEL_BASE" --model_name "$MODEL_NAME" \
  --precision bf16 --quant_text_encoder 1 \
  --prompt_set "$PROMPT_SET" --image_prompt_dir none \
  --num_prompts_per_dimension $NPP --num_videos_per_prompt $NVP \
  --num_prompts_diversity 2 --num_videos_per_prompt_diversity 2 \
  --watermark_method sigmark \
  --width $W --height $H --num_frames $NF --num_steps $STEPS --fps $FPS \
  --guidance_scale $GUID \
  --ch_factor 2 --hw_factor 8 --fr_factor 16 \
  --output_path "$OUT" || { echo "GEN FAILED"; exit 2; }

echo "=== [$(date)] SIGMark GLOBAL 512-bit 2/2: extract (clean, no disturbance) ==="
$PY main.py --mode extract \
  --model_base_path "$MODEL_BASE" --model_name "$MODEL_NAME" \
  --precision bf16 --quant_text_encoder 1 \
  --prompt_set "$PROMPT_SET" --image_prompt_dir none \
  --num_prompts_per_dimension $NPP --num_videos_per_prompt $NVP \
  --num_prompts_diversity 2 --num_videos_per_prompt_diversity 2 \
  --watermark_method sigmark \
  --width $W --height $H --num_frames $NF --num_steps $STEPS --fps $FPS \
  --guidance_scale $GUID \
  --ch_factor 2 --hw_factor 8 --fr_factor 16 \
  --output_path "$OUT" || { echo "EXTRACT FAILED"; exit 3; }

echo "=== [$(date)] SIGMark GLOBAL 512-bit: aggregate per-dimension ==="
$PY summarize_sigmark_semantic.py --video_path "$OUT" \
  || echo "SUMMARIZE FAILED (non-fatal)"

echo "=== [$(date)] SIGMark GLOBAL 512-bit DONE ==="
