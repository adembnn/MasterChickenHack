#!/usr/bin/env bash
# Bootstrap tache 1 sur une machine GPU quelconque (Nebius H200, autre cloud, ou noeud JURECA).
# Ubuntu avec driver NVIDIA deja installe, CUDA 12.4 minimum recommande pour les wheels torch 2.5.1.
# Usage : bash bootstrap_gpu.sh /chemin/de/travail
set -euo pipefail

WORK="${1:-$HOME/eac-t1}"
mkdir -p "$WORK"
cd "$WORK"

echo "== 1. verification du driver"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "== 2. uv et python 3.12"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv -p 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "== 3. kit de depart"
if [ ! -d starter_kit ]; then
  curl -LsSf -o kit.zip "https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task/resolve/main/starter_kit.zip"
  unzip -q kit.zip -d starter_kit
  mv starter_kit/starter_kit/* starter_kit/ 2>/dev/null || true
fi
cd starter_kit

echo "== 4. dependances"
uv pip install -r requirements.txt

echo "== 5. modeles (PixelSeal, TrustMark, StegaStamp)"
if [ ! -f models/pixelseal_card.yaml ]; then
  python download_models.py
fi

echo "== 6. donnees (train 190 Mo, test 379 Mo)"
curl -LsSf -o train.zip "https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task/resolve/main/train.zip"
curl -LsSf -o test.zip "https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task/resolve/main/test.zip"
python -c "from wmloc.data import unzip; unzip('train.zip','release'); unzip('test.zip','release')"

echo "== 7. script de localisation"
# Il est dans le depot EAC, branche yassir/baselines, dossier tracks/t1-watermark.
# Depuis une autre machine : git clone du depot, puis copier le fichier ici.
if [ ! -f decode_localize.py ]; then
  echo "MANQUE decode_localize.py : copie tracks/t1-watermark/decode_localize.py du depot EAC ici." >&2
  exit 1
fi

echo "== 8. calibration rapide sur 30 images, pour voir un mIoU en deux minutes"
python decode_localize.py --data release --calibrate --limit 30

echo "== pret. Suite :"
echo "   python decode_localize.py --data release --calibrate      # seuils sur les 510 images de train"
echo "   python decode_localize.py --data release --out submission.pt"
