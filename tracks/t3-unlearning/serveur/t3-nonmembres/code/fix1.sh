# Relance des voies SDXL dans l'allocation du job 1 (GPU 1 à 3), après correction du cache.
B=/p/scratch/training2666/destors1/MasterChickenHack/unlearning; D=$B/adem/t3-nonmembres; cd $D/code
export HF_HOME=$D/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 T3DIR=$B NMROOT=$D SIMDIR=$D/code TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
PY=$D/.venv/bin/python; L=$D/logs
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
( export CUDA_VISIBLE_DEVICES=1 BS=8; $PY gen_sdxl.py i2i sdxl_i2i_50 0.5 subset > $L/sdxl_i2i_50.log 2>&1; $PY gen_sdxl.py i2i sdxl_i2i_30 0.3 subset > $L/sdxl_i2i_30.log 2>&1; $PY gen_sdxl.py t2i sdxl_t2i 0 subset > $L/sdxl_t2i.log 2>&1 ) &
( export CUDA_VISIBLE_DEVICES=2 BS=8; $PY gen_sdxl.py i2i sdxl_i2i_70 0.7 subset > $L/sdxl_i2i_70.log 2>&1; $PY gen_sdxl.py i2i sdxl_i2i_85 0.85 subset > $L/sdxl_i2i_85.log 2>&1 ) &
( export CUDA_VISIBLE_DEVICES=3; $PY gen_sdxl.py ipa sdxl_ipa_60 0.6 subset > $L/sdxl_ipa_60.log 2>&1 ) &
wait; rm -f $L/hold; echo FIX1_DONE; date
