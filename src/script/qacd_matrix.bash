#!/bin/bash
#
# Full POPE experiment matrix for QACD on LLaVA-1.5-7B, single VM, sequential.
#   {coco, aokvqa} x {random, popular, adversarial} x {vcd_pure, tight, hysteresis} x seeds
#
# RESUMABLE: a cell whose answer file is already complete (record count == gt) is
# skipped, so a Spot preemption just resumes where it left off. Run logged:
#   bash script/qacd_matrix.bash 2>&1 | tee ../matrix.log
#
# Env overrides:
#   datasets="coco aokvqa"  splits="random popular adversarial"
#   methods="vcd_pure savcd tight hysteresis"  seeds="11 21 31"  cd_alpha=1
#     vcd_pure = vanilla VCD (no SAT); savcd = SA-VCD (SAS+SAT); tight/hysteresis = QACD
#   reasoning ablation:  methods="tight_reason hyst_reason"   (QACD + one-sentence REASON)
# After it finishes (or any time):  python eval/qacd_summary.py output/matrix/*.jsonl

export PYTHONPATH=$PYTHONPATH:$(pwd)
set -uo pipefail            # NOT -e: keep going if one cell errors

model_id="llava15_7b"
datasets="${datasets:-coco aokvqa}"
splits="${splits:-random popular adversarial}"
methods="${methods:-vcd_pure savcd tight hysteresis}"
seeds="${seeds:-11 21 31}"
cd_alpha="${cd_alpha:-1}"
image_folder="${image_folder:-./data/POPE/coco/images}"   # val2014: used by BOTH datasets
cuda="${cuda:-0}"
out_root="./output/matrix"
mkdir -p "${out_root}"

# shared QACD knobs (current best); methods differ only in cd-mode + thresholding
qacd_common=(--qacd-region attention --qacd-layer 16 --no-qacd-sink-norm
             --qacd-smooth-sigma 0.8 --qacd-min-region 2
             --qacd-ops "blur,downsample,noise,obscure,r-noise"
             --qacd-prompt adversarial --qacd-icl)

run_cfg () {                # $1 method  $2 dataset  $3 split  $4 seed
  local method="$1" ds="$2" sp="$3" seed="$4"
  local qfile="./data/POPE/${ds}/${ds}_pope_${sp}.jsonl"
  # record count via grep -c (robust to missing trailing newline; gt files lack
  # one, so wc -l would undercount by 1 and never recognize a cell as complete).
  local ntotal; ntotal=$(grep -c . "${qfile}")
  local ans="${out_root}/${method}_${ds}_${sp}_seed${seed}.jsonl"

  if [ -f "${ans}" ] && [ "$(grep -c . "${ans}")" -eq "${ntotal}" ]; then
    echo "[skip] ${method} ${ds} ${sp} seed${seed} (complete, ${ntotal} q)"
  else
    # Do NOT delete a partial file -- pope.py now resumes per-question by
    # skipping qids already present in the answers file. An interrupted cell
    # picks up where it left off rather than restarting from question 0.
    if [ -f "${ans}" ]; then
      ndone=$(grep -c . "${ans}")
      echo "[run ] ${method} ${ds} ${sp} seed${seed} (resume ${ndone}/${ntotal}, $(date +%H:%M:%S))"
    else
      echo "[run ] ${method} ${ds} ${sp} seed${seed} ($(date +%H:%M:%S))"
    fi
    # cd-tau (SAT) is per-method: omit it for vanilla VCD, include 0.5 elsewhere
    local args=(--model-id ${model_id} --image-folder ${image_folder}
                --question-file "${qfile}" --answers-file "${ans}"
                --cd-alpha ${cd_alpha} --cd-beta 0.1
                --seed ${seed} --cuda ${cuda})
    case "${method}" in
      vcd_pure)   args+=(--cd-mode vcd) ;;                                    # vanilla VCD (no SAT)
      savcd)      args+=(--cd-mode selfaug --cd-tau 0.5) ;;                   # SA-VCD (SAS + SAT), inline
      tight)         args+=(--cd-mode qacd --cd-tau 0.5 "${qacd_common[@]}" --qacd-thresh-mode std --qacd-lam 1.0 --qacd-dilate 0) ;;
      hysteresis)    args+=(--cd-mode qacd --cd-tau 0.5 "${qacd_common[@]}" --qacd-thresh-mode hysteresis --qacd-grow-ratio 0.5 --qacd-dilate 0) ;;
      tight_reason)  args+=(--cd-mode qacd --cd-tau 0.5 "${qacd_common[@]}" --qacd-thresh-mode std --qacd-lam 1.0 --qacd-dilate 0 --qacd-reason) ;;     # QACD-tight + Reason
      hyst_reason)   args+=(--cd-mode qacd --cd-tau 0.5 "${qacd_common[@]}" --qacd-thresh-mode hysteresis --qacd-grow-ratio 0.5 --qacd-dilate 0 --qacd-reason) ;;  # QACD-hyst + Reason
      *) echo "[FAIL] unknown method ${method}"; return ;;
    esac
    python eval/pope.py "${args[@]}" || { echo "[FAIL] ${ans} (see error above)"; return; }
  fi
  python eval/eval_pope.py --gt-files "${qfile}" --model-outputs "${ans}" \
    | sed "s/^/    ${method}/"
}

echo "MATRIX: datasets=[${datasets}] splits=[${splits}] methods=[${methods}] seeds=[${seeds}]"
# seed-MAJOR order: complete an entire seed (all datasets/splits/methods) before
# the next seed starts -- so an interrupted/credit-limited run still yields a
# complete single-seed table rather than a lopsided partial.
for s in ${seeds}; do
  for ds in ${datasets}; do
    for sp in ${splits}; do
      for m in ${methods}; do
        run_cfg "${m}" "${ds}" "${sp}" "${s}"
      done
    done
  done
done

echo ""
echo "==================== matrix complete ===================="
echo "Tabulated summary (mean +/- std over seeds):"
python eval/qacd_summary.py "${out_root}"/*.jsonl
