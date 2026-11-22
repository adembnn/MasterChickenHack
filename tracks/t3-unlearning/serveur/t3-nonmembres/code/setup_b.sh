cd ~/eac/t3/t3-nonmembres
~/.local/bin/uv venv --python 3.11 .venv
VIRTUAL_ENV=$PWD/.venv ~/.local/bin/uv pip install torch torchvision timm diffusers transformers accelerate sentencepiece protobuf peft einops pillow numpy 2>&1 | tail -3
.venv/bin/python -c "import torch,diffusers,transformers,timm;print(\"ENV OK\",torch.__version__,torch.cuda.is_available(),diffusers.__version__)"
.venv/bin/python code/dl_noflux.py 2>&1 | grep -E "^OK|^ERR"
echo SETUP_DONE
