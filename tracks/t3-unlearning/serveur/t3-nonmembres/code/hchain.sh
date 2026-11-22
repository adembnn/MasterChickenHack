# usage : hchain.sh T3DIR NMROOT "mode tag force quoi graine" ...   (H200 ; génération puis mesure, reprise possible, 4 essais par étape)
export T3DIR=$1 NMROOT=$2 SIMDIR=$HOME/eac/t3/t3-nonmembres/code BS=${BS:-8} NOHOLD=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4; shift 2
cd $HOME/eac/t3/t3-nonmembres/code; PY=../.venv/bin/python; mkdir -p $NMROOT/img $NMROOT/meas $NMROOT/logs
[ -f $NMROOT/meas/subset.npy ] || $PY prep.py > $NMROOT/logs/prep.log 2>&1
[ -f $NMROOT/meas/captions.json ] || { $PY cap.py 0 1 > $NMROOT/logs/cap.log 2>&1; $PY cap.py merge 1 >> $NMROOT/logs/prep.log 2>&1; }
[ -f $NMROOT/meas/eval_forget.json ] || $PY evalnm.py forget > $NMROOT/logs/eval_forget.log 2>&1
for spec in "$@"; do set -- $spec
  for try in 1 2 3 4; do [ -f $NMROOT/img/$2.npz ] && break; $PY gen_sdxl.py $1 $2 $3 $4 $5 > $NMROOT/logs/$2.log 2>&1 || sleep 45; done
  for try in 1 2 3 4; do [ -f $NMROOT/meas/eval_$2.json ] && break; $PY evalnm.py $2 > $NMROOT/logs/eval_$2.log 2>&1 || sleep 30; done
done; echo HCHAIN_DONE; date
