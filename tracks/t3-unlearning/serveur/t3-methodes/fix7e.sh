cd ~/eac/t3
ls -la .venv 2>&1 | head -3; ls /home/hermes/eac/kit/.venv/bin/python*
[ -x .venv/bin/python ] || { rm -rf .venv; ln -s /home/hermes/eac/kit/.venv .venv; }
ls -la .venv/bin/python
cd t3-methodes; PY=../.venv/bin/python
run() { N=$1; T=$2; shift 2
  (nohup $PY code/train.py --anchor mix --wce 1 "$@" --name $N --tgt $T --fset all > logs/$N.log 2>&1 &); echo "lance $N"; }
run PGhoB2_f60_l814c HO60 --lam 3 --steps 1500 --blocks 2
run PGhoB3_f60_l814c HO60 --lam 3 --steps 1500 --blocks 3
run PGhoB6_f60_l814c HO60 --lam 3 --steps 1500 --blocks 6
run PGhoL2a_f60_l814c HO60 --lam 3 --steps 1500 --l2sp 0.01
run PGhoL2b_f60_l814c HO60 --lam 3 --steps 1500 --l2sp 0.1
run PGhoBA_f60_l814c HO60 --lam 3 --steps 1500 --ba 192
run PGhoLR5_f60_l814c HO60 --lam 3 --steps 3000 --lr 5e-5
run PGho6s3k_f60_l814c HO60 --lam 6 --steps 3000
sleep 50; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; for f in logs/PGho?*_f60_l814c.log; do echo "$f $(tail -c 200 $f | tr '\r' '\n' | tail -1 | cut -c1-90)"; done; pgrep -u adem -fc "finish.sh"
