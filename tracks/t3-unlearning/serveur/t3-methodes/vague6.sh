cd ~/eac/t3/t3-methodes
PY=../.venv/bin/python; P85=0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise
[ -s cache/tgt_HO80.pt ] || $PY code/mk_tgt.py HO80 --pool $P85 --f 0.80 --llo 8 --lhi 14 --softn 0 --wsrc orig > logs/mk_HO80.log 2>&1
run() { N=$1; T=$2; shift 2
  pgrep -u adem -f -- "--name $N " > /dev/null && { echo "deja en cours $N"; return; }
  (nohup $PY code/train.py --anchor mix --wce 1 "$@" --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &)
  echo "lance $N"
}
run PGho_f70_l814c HO70 --lam 3 --steps 1500
run PGho_f80_l814c HO80 --lam 3 --steps 1500
run PGho6_f60_l814c HO60 --lam 6 --steps 1500
run PGho10_f60_l814c HO60 --lam 10 --steps 1500
run PGho3k_f60_l814c HO60 --lam 3 --steps 3000
run PGho6_f70_l814c HO70 --lam 6 --steps 1500
$PY ~/eac/t3/orchestrateur/mk_g10.py candidates/PG_T85_f70_l814c.pt candidates/G10_f70.pt | tail -1
sleep 25; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; tail -c 200 logs/PGho_f70_l814c.log | tr '\r' '\n' | tail -1 | cut -c1-120
