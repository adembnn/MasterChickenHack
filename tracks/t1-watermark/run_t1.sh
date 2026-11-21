#!/usr/bin/env bash
# Calibration T1 complete sur la H200, dans l'ordre : reference all-zeros, canevas, crop.
set -x
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/eac/kit"
# shellcheck disable=SC1091
source .venv/bin/activate

date -u
python submission_template.py --data release --out sub_zeros.pt 2>&1 | tail -5

python decode_localize.py --data release --calibrate --decode-mode canvas \
  --calib-out calib_canvas.json --save-maps maps_canvas
date -u
python decode_localize.py --data release --calibrate --decode-mode crop \
  --calib-out calib_crop.json
date -u
echo CALIB_DONE
