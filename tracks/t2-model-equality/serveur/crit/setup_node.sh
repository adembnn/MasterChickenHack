#!/bin/bash
# Installe l'environnement T2 (memes versions que h200) sur un noeud libre.
set -e
mkdir -p ~/.local/bin /home/adem/eac/t2/crit /home/adem/eac/t2/hf
chmod +x ~/.local/bin/uv
cd /home/adem/eac/t2
[ -x .venv/bin/python ] || ~/.local/bin/uv venv .venv --python 3.12 > setup.log 2>&1
~/.local/bin/uv pip install --python .venv/bin/python -r crit/req_t2.txt >> setup.log 2>&1
.venv/bin/python -c "import torch, transformers, peft; print('env ok', torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_available())" >> setup.log 2>&1
tail -1 setup.log
