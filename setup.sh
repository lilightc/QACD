#!/usr/bin/env bash
#
# QACD VM setup script.
# Target: GCP g2-standard-12 (1× L4) on Deep Learning VM image (common-cu124).
# Run after SSHing into the VM:
#   bash setup.sh
#
# Idempotent: safe to re-run after partial failures.
set -euo pipefail

REPO_URL="https://github.com/lilightc/QACD.git"
REPO_DIR="${HOME}/QACD"
ENV_NAME="qacd"
LLAVA_DIR="${REPO_DIR}/src/models/llava/llava-v1.5-7b"
COCO_DIR="${REPO_DIR}/src/data/POPE/coco/images"

log()  { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[warn]\033[0m  %s\n' "$*"; }

# --- 1. Sanity check: GPU + driver -------------------------------------------
log "Checking NVIDIA driver and GPU..."
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found. Are you on the Deep Learning VM image?"
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# --- 2. Clone repo -----------------------------------------------------------
if [ ! -d "${REPO_DIR}/.git" ]; then
  log "Cloning QACD repo..."
  git clone "${REPO_URL}" "${REPO_DIR}"
else
  log "Repo already present at ${REPO_DIR}; pulling latest."
  git -C "${REPO_DIR}" pull --ff-only
fi

# --- 3. Conda env ------------------------------------------------------------
if ! command -v conda >/dev/null; then
  echo "conda not found. The Deep Learning VM should have it; check PATH."
  exit 1
fi
# Make `conda activate` work inside this script.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}\b"; then
  log "Conda env '${ENV_NAME}' already exists; skipping creation."
else
  log "Creating conda env '${ENV_NAME}' from environment.yml (this takes 10-20 min)..."
  conda env create -f "${REPO_DIR}/environment.yml"
fi
conda activate "${ENV_NAME}"

# --- 4. Download LLaVA-1.5-7B checkpoint -------------------------------------
if [ -f "${LLAVA_DIR}/config.json" ]; then
  log "LLaVA-1.5-7B already downloaded at ${LLAVA_DIR}; skipping."
else
  log "Downloading LLaVA-1.5-7B from HuggingFace (~14 GB, 10-20 min)..."
  pip install -q "huggingface_hub[cli]"
  mkdir -p "${LLAVA_DIR}"
  huggingface-cli download liuhaotian/llava-v1.5-7b \
    --local-dir "${LLAVA_DIR}" \
    --local-dir-use-symlinks False
fi

# --- 5. Download COCO val2014 (POPE images) ----------------------------------
if [ -d "${COCO_DIR}" ] && [ "$(ls -A "${COCO_DIR}" | wc -l)" -gt 0 ]; then
  log "COCO val2014 already present at ${COCO_DIR}; skipping."
else
  log "Downloading COCO val2014 (~6 GB zip, ~12 GB unpacked)..."
  mkdir -p "${COCO_DIR}"
  TMP_ZIP="$(mktemp -t coco-val2014-XXXXXX.zip)"
  curl -L -o "${TMP_ZIP}" http://images.cocodataset.org/zips/val2014.zip
  unzip -q "${TMP_ZIP}" -d "${COCO_DIR}/.."
  rm -f "${TMP_ZIP}"
  # POPE expects images directly under coco/images; val2014 unzips to a val2014/ subdir
  if [ -d "${COCO_DIR}/../val2014" ] && [ ! -L "${COCO_DIR}" ]; then
    rmdir "${COCO_DIR}" || true
    ln -s val2014 "${COCO_DIR}"
  fi
fi

# --- 6. Smoke test: GPU visible to torch -------------------------------------
log "Verifying torch can see the GPU..."
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() == False"
print(f"  torch       : {torch.__version__}")
print(f"  CUDA build  : {torch.version.cuda}")
print(f"  GPU         : {torch.cuda.get_device_name(0)}")
print(f"  GPU memory  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PY

log "Setup complete."
cat <<EOF

Next steps:
  cd ${REPO_DIR}/src
  conda activate ${ENV_NAME}

  # Edit the image_folder in src/script/pope.bash to point at:
  #   ${COCO_DIR}
  # Then sanity-run a single POPE split:
  bash script/pope.bash

EOF
