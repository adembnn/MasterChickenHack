# usage : simchain.sh members|nonmembers GPU   (calibration dans le simulateur, reprise possible)
MODE=$1; export CUDA_VISIBLE_DEVICES=$2 BS=8 NOHOLD=1
B=/p/scratch/training2666/destors1/MasterChickenHack/unlearning; D=$B/adem/t3-nonmembres; cd $D/code
SIM=$B/adem/t3-simulateur; SC=$D/simcal/$MODE; PY=$D/.venv/bin/python; L=$D/logs/simcal_$MODE; mkdir -p $SC/t3 $SC/root/img $SC/root/meas $L
export HF_HOME=$D/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SIMDIR=$D/code TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
[ -f $SC/t3/forget_images.npz ] || SIMDIR=$D/code $PY simcal_prep.py $SIM $SIM/runs/A/gold_s2.pt $SC/t3 $MODE > $L/prep0.log 2>&1
export T3DIR=$SC/t3 NMROOT=$SC/root
[ -f $SC/root/meas/subset.npy ] || $PY prep.py > $L/prep.log 2>&1
[ -f $SC/root/meas/captions.json ] || { $PY cap.py 0 1 > $L/cap.log 2>&1; $PY cap.py merge 1 >> $L/prep.log 2>&1; }
$PY evalnm.py forget > $L/eval_forget.log 2>&1
for spec in "i2i sdxl_i2i_70 0.7" "i2i sdxl_i2i_50 0.5" "i2i sdxl_i2i_85 0.85" "ipa sdxl_ipa_60 0.6" "i2i sdxl_i2i_60 0.6" "t2i sdxl_t2i 0" "i2i sdxl_i2i_30 0.3"; do
  set -- $spec; [ -f $SC/root/img/$2.npz ] || $PY gen_sdxl.py $1 $2 $3 subset > $L/$2.log 2>&1
  [ -f $SC/root/meas/eval_$2.json ] || $PY evalnm.py $2 > $L/eval_$2.log 2>&1
done
echo SIMCHAIN_DONE $MODE; date
