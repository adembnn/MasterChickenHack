#!/usr/bin/env bash
# Ablations T1 sur le train complet, en mode canevas. Chaque ligne imprime le mIoU global.
set -x
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/eac/kit"
# shellcheck disable=SC1091
source .venv/bin/activate

run () {
  name="$1"; shift
  echo "### $name"
  python decode_localize.py --data release --calibrate --decode-mode canvas \
    --calib-out "calib_$name.json" "$@" 2>/dev/null | grep -E "train mIoU|pixelseal|trustmark|stegastamp"
}

run wpos0      --w-pos 0
run wpos10     --w-pos 1.0
run win192s48  --win 192 --stride 48
run win320s80  --win 320 --stride 80
run nowindow   --no-window
echo ABLATIONS_DONE
