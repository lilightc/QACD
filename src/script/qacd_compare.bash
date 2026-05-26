#!/bin/bash
#
# Compare two QACD region configs on the SAME POPE split, back to back:
#   tight       : std threshold, lam=1.0, no dilation        (small, salient-part region)
#   hysteresis  : seed at mean+lam*std, grow to object extent (covers object, stops at bg)
#
# Everything else is held fixed (sink-off, smooth 0.8, constrained ops), so the
# only difference is the thresholding -> a clean A/B on region extraction.
#
# From src/:   bash script/qacd_compare.bash
# Env overrides:
#   type=adversarial   bash script/qacd_compare.bash   # the split where CD actually shows
#   limit=300          bash script/qacd_compare.bash   # quick partial run (0 = full split)
#   dataset_name=coco | aokvqa ; seed=55

export PYTHONPATH=$PYTHONPATH:$(pwd)
set -e

model_id="llava15_7b"
dataset_name="${dataset_name:-coco}"
type="${type:-popular}"
limit="${limit:-0}"                 # 0 = full split; e.g. 300 for a quick check
seed="${seed:-55}"
image_folder="${image_folder:-./data/POPE/coco/images}"
cuda="${cuda:-0}"
qfile="./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.jsonl"
out_root="./output/compare"

# shared knobs (current best); only the threshold mode differs per config
common=(--model-id ${model_id}
        --image-folder ${image_folder}
        --question-file ${qfile}
        --cd-mode qacd
        --cd-alpha 1 --cd-beta 0.1 --cd-tau 0.5
        --qacd-region attention --qacd-layer 16
        --qacd-smooth-sigma 0.8 --qacd-min-region 2
        --no-qacd-sink-norm
        --qacd-ops "blur,downsample,noise,obscure,r-noise"
        --qacd-prompt adversarial --qacd-icl
        --seed ${seed} --cuda ${cuda})
[ "${limit}" -gt 0 ] && common+=(--limit ${limit})

run_one () {            # $1 = config name ; $2.. = config-specific flags
  local name="$1"; shift
  local out="${out_root}/${name}_${dataset_name}_${type}_seed${seed}.jsonl"
  mkdir -p "${out_root}"; rm -f "${out}"
  echo ""
  echo "################  ${name}  ################"
  python eval/pope.py "${common[@]}" "$@" --answers-file "${out}"
  echo "----  ${name}: accuracy / precision / recall / F1  ----"
  python eval/eval_pope.py --gt-files "${qfile}" --model-outputs "${out}"
  echo "----  ${name}: region + op distribution  ----"
  python eval/qacd_stats.py "${out}"
}

echo "Comparing region configs on ${dataset_name} ${type} (limit=${limit:-full}, seed ${seed})"
run_one "tight"       --qacd-thresh-mode std         --qacd-lam 1.0  --qacd-dilate 0
run_one "hysteresis"  --qacd-thresh-mode hysteresis  --qacd-grow-ratio 0.5 --qacd-dilate 0

echo ""
echo "================================================================"
echo "Done. Compare the F1 lines above (tight vs hysteresis)."
echo "Answers: ${out_root}/{tight,hysteresis}_${dataset_name}_${type}_seed${seed}.jsonl"
echo "For the verdict that matters, also run with:  type=adversarial bash script/qacd_compare.bash"
