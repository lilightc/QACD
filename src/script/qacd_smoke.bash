#!/bin/bash
#
# QACD smoke test: a handful of POPE questions with intermediate-result dumps.
# Use this to sanity-check the planner + attention region before a full run.
# From src/:   bash script/qacd_smoke.bash
#
# Inspect afterwards:
#   output/qacd_debug/<N>.png        side-by-side: original | region overlay | corrupted
#   output/qacd_debug/recipes.jsonl  planner target/op/intensity, region, mask coverage

export PYTHONPATH=$PYTHONPATH:$(pwd)
set -e

model_id="llava15_7b"
cd_mode="${cd_mode:-qacd}"                # "qacd" | "no_vcd" (baseline) | "vcd"
limit="${limit:-36}"                      # 36 questions = ~6 POPE images
qacd_region="${qacd_region:-attention}"
qacd_layer="${qacd_layer:-16}"
qacd_thresh_mode="${qacd_thresh_mode:-std}"   # std | hysteresis
qacd_grow_ratio="${qacd_grow_ratio:-0.5}"   # hysteresis low/high ratio (lower=grows more)
qacd_lam="${qacd_lam:-0.5}"               # seed/threshold mean+lam*std (lower=broader)
qacd_sink_norm="${qacd_sink_norm:-0}"     # 0=off (default); 1=baseline subtraction
qacd_smooth_sigma="${qacd_smooth_sigma:-0.8}"
qacd_min_region="${qacd_min_region:-2}"
qacd_dilate="${qacd_dilate:-1}"           # dilate N grid cells (coverage)
qacd_prompt="${qacd_prompt:-adversarial}"

dataset_name="coco"
type="popular"
image_folder=./data/POPE/coco/images   # symlink created by setup.sh (-> val2014)
debug_dir=./output/smoke_${cd_mode}     # tagged by mode so runs don't clobber
answer_file=${debug_dir}/answers.jsonl
cuda=0

# fresh debug dir each run (recipes.jsonl/answers.jsonl are append-mode)
rm -rf "${debug_dir}"

python eval/pope.py \
  --model-id ${model_id} \
  --image-folder ${image_folder} \
  --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.jsonl \
  --answers-file ${answer_file} \
  --cd-mode ${cd_mode} \
  --cd-alpha 1 --cd-beta 0.1 --cd-tau 0.5 \
  --qacd-region ${qacd_region} \
  --qacd-layer ${qacd_layer} \
  --qacd-thresh-mode ${qacd_thresh_mode} \
  --qacd-grow-ratio ${qacd_grow_ratio} \
  --qacd-lam ${qacd_lam} \
  $([ "${qacd_sink_norm}" = "1" ] && echo --qacd-sink-norm || echo --no-qacd-sink-norm) \
  --qacd-smooth-sigma ${qacd_smooth_sigma} \
  --qacd-min-region ${qacd_min_region} \
  --qacd-dilate ${qacd_dilate} \
  --qacd-prompt ${qacd_prompt} \
  --qacd-debug-dir ${debug_dir} \
  --limit ${limit} \
  --seed 11 \
  --cuda ${cuda}

echo ""
echo "Smoke test done. Inspect:"
echo "  ${debug_dir}/*.png"
echo "  ${debug_dir}/recipes.jsonl"
