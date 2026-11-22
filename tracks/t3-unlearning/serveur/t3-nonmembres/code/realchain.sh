# usage : realchain.sh GPU FICHIER_A_ATTENDRE "mode tag force quoi graine" ...   (variations des images réelles, reprise possible)
export CUDA_VISIBLE_DEVICES=$1 BS=8 NOHOLD=1; WAITF=$2; shift 2
B=/p/scratch/training2666/destors1/MasterChickenHack/unlearning; D=$B/adem/t3-nonmembres; cd $D/code; PY=$D/.venv/bin/python; L=$D/logs
export HF_HOME=$D/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 T3DIR=$B NMROOT=$D SIMDIR=$D/code TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
while [ "$WAITF" != "-" ] && [ ! -f "$WAITF" ]; do sleep 15; done
for spec in "$@"; do set -- $spec; $PY gen_sdxl.py $1 $2 $3 $4 $5 > $L/$2.log 2>&1; $PY evalnm.py $2 > $L/eval_$2.log 2>&1; done
echo REALCHAIN_DONE; date
