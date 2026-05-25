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
limit="${limit:-8}"
qacd_region="${qacd_region:-attention}"
qacd_layer="${qacd_layer:-16}"
qacd_lam="${qacd_lam:-0.5}"
qacd_prompt="${qacd_prompt:-adversarial}"

dataset_name="coco"
type="popular"
image_folder=./data/POPE/coco/images   # symlink created by setup.sh (-> val2014)
debug_dir=./output/qacd_debug
answer_file=./output/qacd_debug/answers.jsonl
cuda=0

# fresh debug dir each run (recipes.jsonl/answers.jsonl are append-mode)
rm -rf "${debug_dir}"

python eval/pope.py \
  --model-id ${model_id} \
  --image-folder ${image_folder} \
  --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.jsonl \
  --answers-file ${answer_file} \
  --cd-mode qacd \
  --cd-alpha 1 --cd-beta 0.1 --cd-tau 0.5 \
  --qacd-region ${qacd_region} \
  --qacd-layer ${qacd_layer} \
  --qacd-lam ${qacd_lam} \
  --qacd-prompt ${qacd_prompt} \
  --qacd-debug-dir ${debug_dir} \
  --limit ${limit} \
  --seed 11 \
  --cuda ${cuda}

echo ""
echo "Smoke test done. Inspect:"
echo "  ${debug_dir}/*.png"
echo "  ${debug_dir}/recipes.jsonl"
