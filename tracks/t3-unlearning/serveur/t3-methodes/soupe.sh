cd ~/eac/t3/t3-methodes; PY=../.venv/bin/python
ls cache | grep "RDt24r"; head -c 600 logs/PG_RDt24r_70_l1016c.log 2>/dev/null | tr '\r' '\n' | head -1 | cut -c1-520
T=$(ls cache | grep "tgt_RDt24r_70_l1016" | head -1 | sed 's/^tgt_//; s/\.pt$//'); echo "cible=$T"
[ -z "$T" ] && exit 0
for s in 1 2 3 4; do
  (nohup $PY code/train.py --anchor mix --wce 1 --lam 3 --steps 1500 --seed $s --name SOUPs${s} --tgt $T --fset all > logs/SOUPs$s.log 2>&1 &)
done
sleep 30; nvidia-smi --query-gpu=memory.used --format=csv,noheader; tail -c 150 logs/SOUPs1.log | tr '\r' '\n' | tail -1 | cut -c1-100; ls deltas | head -3; $PY - <<'PYE'
import torch,glob
f=sorted(glob.glob('deltas/*.pt'))[0]; d=torch.load(f,map_location='cpu'); print(f, type(d).__name__, (list(d.keys())[:4] if isinstance(d,dict) else ''), len(d) if hasattr(d,'__len__') else '')
PYE
