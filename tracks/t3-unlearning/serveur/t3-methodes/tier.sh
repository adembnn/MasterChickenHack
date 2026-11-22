cd ~/eac/t3/t3-methodes; PY=../.venv/bin/python; P85=0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise
R=/home/adem/eac/t3/t3-redistill/targets
grep -n "tier2" code/mk_tgt.py | cut -c1-230 | head -8
mk() { T=$1; f=$2; t2=$3
  [ -s cache/tgt_$T.pt ] || $PY code/mk_tgt.py $T --pool $P85 --f $f --llo 10 --lhi 16.5 --tier2 $t2 --rank ext:$R/redistill_k4m3.npy --wsrc ext:$R/redistill_tau2.4.npy --wext 1.0 2>&1 | tail -1 | python3 -c "import sys,json; l=sys.stdin.read(); i=l.find('{'); d=json.loads(l[i:]); print({k:(round(v,3) if isinstance(v,float) else v) for k,v in d.items() if k!='args'})" 2>&1 | cut -c1-200
  [ -s cache/tgt_$T.pt ] || { echo "cible $T absente"; return; }
  (nohup $PY code/train.py --anchor mix --wce 1 --lam 3 --steps 1500 --name PG_${T}c --tgt $T --fset all > logs/PG_${T}c.log 2>&1 &)
  (nohup bash code/finish.sh PG_${T}c $T > logs/finish_PG_${T}c.log 2>&1 &); echo "lance PG_${T}c"; }
mk RDt24r_60t20a 0.60 0.20,1,3
mk RDt24r_65t15a 0.65 0.15,1,3
mk RDt24r_70t10a 0.70 0.10,1,3
mk RDt24r_60t20b 0.60 0.20,3,8
mk RDt24r_70t15b 0.70 0.15,2,6
sleep 25; nvidia-smi --query-gpu=memory.used --format=csv,noheader
