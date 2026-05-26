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

# cd_alpha = contrastive strength (the over-suppression lever); override via env
cd_alpha="${cd_alpha:-1}"
# Baseline = VCD: QACD = VCD + query-aware grounding, so QACD-vs-VCD isolates the
# contribution. Greedy is omitted (VCD>greedy is established in the literature).
# Add it back if needed via baselines="no_vcd vcd".
baselines="${baselines:-vcd}"

# shared knobs (current best); cd-mode and the threshold mode differ per config
common=(--model-id ${model_id}
        --image-folder ${image_folder}
        --question-file ${qfile}
        --cd-alpha ${cd_alpha} --cd-beta 0.1 --cd-tau 0.5
        --qacd-region attention --qacd-layer 16
        --qacd-smooth-sigma 0.8 --qacd-min-region 2
        --no-qacd-sink-norm
        --qacd-ops "blur,downsample,noise,obscure,r-noise"
        --qacd-prompt adversarial --qacd-icl
        --seed ${seed} --cuda ${cuda})
[ "${limit}" -gt 0 ] && common+=(--limit ${limit})

run_one () {            # $1 = config name ; $2 = cd_mode ; $3.. = extra flags
  local name="$1" mode="$2"; shift 2
  local out="${out_root}/${name}_${dataset_name}_${type}_seed${seed}.jsonl"
  mkdir -p "${out_root}"; rm -f "${out}"
  echo ""
  echo "################  ${name}  (cd_mode=${mode})  ################"
  python eval/pope.py "${common[@]}" --cd-mode "${mode}" "$@" --answers-file "${out}"
  echo "----  ${name}: metrics  ----"
  python eval/eval_pope.py --gt-files "${qfile}" --model-outputs "${out}"
  [ "${mode}" = "qacd" ] && { echo "----  ${name}: region + op distribution  ----"; python eval/qacd_stats.py "${out}"; }
}

echo "Comparing on ${dataset_name} ${type} (limit=${limit:-full}, seed ${seed}, cd_alpha=${cd_alpha})"
echo "baselines: ${baselines}"
for b in ${baselines}; do run_one "${b}" "${b}"; done       # greedy / vcd reference(s)
run_one "tight"       qacd --qacd-thresh-mode std         --qacd-lam 1.0  --qacd-dilate 0
run_one "hysteresis"  qacd --qacd-thresh-mode hysteresis  --qacd-grow-ratio 0.5 --qacd-dilate 0

echo ""
echo "================================================================"
echo "Compare:  vcd  vs  tight/hysteresis(qacd)"
echo "  QACD vs VCD = the contribution (targeted vs global corruption)"
echo "  recall<<precision or yes-prop<<50%% => over-suppression; tune cd_alpha"
echo "Answers under: ${out_root}/"
echo "For the verdict that matters, also run with:  type=adversarial bash script/qacd_compare.bash"
