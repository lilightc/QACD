#!/bin/bash
#
# Full POPE evaluation for QACD -> accuracy / precision / recall / F1 per split.
# From src/:   bash script/qacd_eval.bash
#
# All QACD knobs are env-overridable, so you run ablations by setting vars:
#   qacd_region=full   bash script/qacd_eval.bash     # global corruption (no grounding)
#   qacd_sink_norm=0   bash script/qacd_eval.bash     # raw attention
#   qacd_thresh_mode=std qacd_lam=0.5 bash ...         # fixed-fraction threshold
#   types=popular      bash script/qacd_eval.bash     # one split only (faster)
#   seeds="11 21 31"   bash script/qacd_eval.bash     # average over seeds
#
# Output answers land under output/eval/qacd/<tag>/ ; the per-question 'qacd'
# field is kept, so `python eval/qacd_stats.py <those files>` still works.

export PYTHONPATH=$PYTHONPATH:$(pwd)
set -e

model_id="llava15_7b"
dataset_name="${dataset_name:-coco}"               # coco | aokvqa (both use val2014 images)
types="${types:-popular random adversarial}"       # POPE sampling splits
seeds="${seeds:-55}"                               # space-separated; add more to average
image_folder="${image_folder:-./data/POPE/coco/images}"
cuda="${cuda:-0}"

# --- QACD config (env-overridable; defaults = current best) ---
qacd_region="${qacd_region:-attention}"            # attention | full | center
qacd_layer="${qacd_layer:-16}"
qacd_thresh_mode="${qacd_thresh_mode:-otsu}"       # otsu | std
qacd_lam="${qacd_lam:-1.0}"
qacd_sink_norm="${qacd_sink_norm:-1}"
qacd_smooth_sigma="${qacd_smooth_sigma:-0.6}"
qacd_min_region="${qacd_min_region:-2}"
qacd_dilate="${qacd_dilate:-0}"
qacd_prompt="${qacd_prompt:-adversarial}"
qacd_icl="${qacd_icl:-1}"

[ "${qacd_sink_norm}" = "1" ] && sink_flag="--qacd-sink-norm" || sink_flag="--no-qacd-sink-norm"
[ "${qacd_icl}" = "1" ] && icl_flag="--qacd-icl" || icl_flag="--no-qacd-icl"

# tag output by config so ablation runs never clobber each other
tag="${qacd_region}_${qacd_thresh_mode}_sink${qacd_sink_norm}_L${qacd_layer}_icl${qacd_icl}"
out_dir="./output/eval/qacd/${tag}"
mkdir -p "${out_dir}"
echo "QACD eval config: ${tag}"
echo "splits: ${dataset_name} [${types}] | seeds: ${seeds}"

for type in ${types}; do
  for seed in ${seeds}; do
    ans="${out_dir}/${dataset_name}_${type}_seed${seed}.jsonl"
    rm -f "${ans}"   # pope.py appends; start fresh so re-runs don't duplicate

    echo ""
    echo "================ ${dataset_name} ${type} | seed ${seed} ================"
    python eval/pope.py \
      --model-id ${model_id} \
      --image-folder ${image_folder} \
      --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.jsonl \
      --answers-file ${ans} \
      --cd-mode qacd \
      --cd-alpha 1 --cd-beta 0.1 --cd-tau 0.5 \
      --qacd-region ${qacd_region} \
      --qacd-layer ${qacd_layer} \
      --qacd-thresh-mode ${qacd_thresh_mode} \
      --qacd-lam ${qacd_lam} \
      ${sink_flag} \
      --qacd-smooth-sigma ${qacd_smooth_sigma} \
      --qacd-min-region ${qacd_min_region} \
      --qacd-dilate ${qacd_dilate} \
      --qacd-prompt ${qacd_prompt} \
      ${icl_flag} \
      --seed ${seed} \
      --cuda ${cuda}

    echo "---- metrics: ${dataset_name} ${type} seed ${seed} ----"
    python eval/eval_pope.py \
      --gt-files ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.jsonl \
      --model-outputs ${ans}
  done
done

echo ""
echo "Done. Answers + per-question QACD recipes under: ${out_dir}"
echo "Aggregate recipe/fallback stats:  python eval/qacd_stats.py '${out_dir}/*.jsonl'"
