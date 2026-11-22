cd ~/eac/t3/t3-methodes
PY=/home/hermes/eac/kit/.venv/bin/python
ls -la cache/tgt_T85_f30_l814.pt cache/tgt_T85_f40_l814.pt ../forget_images.npz ../model.pt 2>&1 | awk '{print $5, $9}'
mkdir -p logs deltas candidates
for N in f30 f40; do
  (nohup $PY code/train.py --anchor mix --lam 3 --steps 1500 --wce 1 --name PG_T85_${N}_l814c --tgt T85_${N}_l814 --fset all > logs/PG_T85_${N}_l814c.log 2>&1 &)
done
sleep 45; tail -3 logs/PG_T85_f30_l814c.log; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; cat code/finish.sh | head -30
