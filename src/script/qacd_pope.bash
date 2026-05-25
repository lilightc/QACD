#!/bin/bash
#
# QACD on POPE. Edit `image_folder`, then run from src/:  bash script/qacd_pope.bash
#
# Ablation knobs (override on the command line, e.g. `qacd_region=center bash ...`):
#   qacd_region    : attention | center | full   (Stage-2 grounding ablation)
#   qacd_intensity : 0=planner-chosen, or 1/2/3   (intensity ablation)
#   qacd_ops       : ""=all, or e.g. "blur,noise,desat"  (operation-set ablation)
#   qacd_prompt    : adversarial | neutral        (planner-prompt ablation)
#   qacd_layer     : LLM layer for attention      (layer sweep)
#   qacd_lam       : mask threshold mean+lam*std

export PYTHONPATH=$PYTHONPATH:$(pwd)
set -e

model_id="llava15_7b"
cd_mode="qacd"
cd_alpha=1
cd_beta=0.1
cd_tau=0.5

# --- QACD knobs (env-overridable) ---
qacd_region="${qacd_region:-attention}"
qacd_intensity="${qacd_intensity:-0}"
qacd_ops="${qacd_ops:-}"
qacd_prompt="${qacd_prompt:-adversarial}"
qacd_layer="${qacd_layer:-16}"
qacd_lam="${qacd_lam:-0.5}"
qacd_icl="${qacd_icl:-1}"          # 1=few-shot (default), 0=zero-shot ablation

# --- data ---
dataset_name="coco"        # "coco" | "aokvqa"
type="popular"             # "popular" | "random" | "adversarial"
image_folder=./data/POPE/coco/images   # symlink created by setup.sh (-> val2014)
seeds=(11 21 31)           # fewer seeds than baselines; bump for the final table
cuda=0

# tag output by the ablation setting so runs don't collide
[ "${qacd_icl}" = "1" ] && icl_flag="--qacd-icl" || icl_flag="--no-qacd-icl"
tag="${qacd_region}_int${qacd_intensity}_${qacd_prompt}_L${qacd_layer}_icl${qacd_icl}"

for seed in "${seeds[@]}"; do
  answer_file=./output/${cd_mode}/pope/${tag}/${model_id}_${dataset_name}_${type}_seed${seed}.jsonl

  python eval/pope.py \
    --model-id ${model_id} \
    --image-folder ${image_folder} \
    --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.jsonl \
    --answers-file ${answer_file} \
    --cd-mode ${cd_mode} \
    --cd-alpha ${cd_alpha} \
    --cd-beta ${cd_beta} \
    --cd-tau ${cd_tau} \
    --qacd-region ${qacd_region} \
    --qacd-intensity ${qacd_intensity} \
    --qacd-ops "${qacd_ops}" \
    --qacd-prompt ${qacd_prompt} \
    --qacd-layer ${qacd_layer} \
    --qacd-lam ${qacd_lam} \
    ${icl_flag} \
    --seed ${seed} \
    --cuda ${cuda}

  python eval/eval_pope.py \
    --gt-files ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.jsonl \
    --model-outputs ${answer_file}
done
