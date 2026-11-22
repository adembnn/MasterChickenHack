cd ~/eac/t3/t3-methodes; PY=../.venv/bin/python
nvidia-smi --query-gpu=memory.used --format=csv,noheader; ls candidates | head -20
ls -la cache/tgt_T85_f70_l814.pt
run() { N=$1; T=$2; shift 2
  (nohup $PY code/train.py --anchor mix --wce 1 "$@" --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &); echo "lance $N"; }
T=T85_f70_l814
run PGseed1_f70_l814c $T --lam 3 --steps 1500 --seed 1
run PGB2_f70_l814c $T --lam 3 --steps 1500 --blocks 2
run PGB3_f70_l814c $T --lam 3 --steps 1500 --blocks 3
run PGB6_f70_l814c $T --lam 3 --steps 1500 --blocks 6
run PGL2_f70_l814c $T --lam 3 --steps 1500 --l2sp 0.01
run PGBA_f70_l814c $T --lam 3 --steps 1500 --ba 192
run PGlam6_f70_l814c $T --lam 6 --steps 1500
run PGs3k_f70_l814c $T --lam 3 --steps 3000
run PGlr5_f70_l814c $T --lam 3 --steps 3000 --lr 5e-5
run PGwcev2_f70_l814c $T --lam 3 --steps 1500 --wcev 2
run PGwcev3_f70_l814c $T --lam 3 --steps 1500 --wcev 3
sleep 40; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; tail -c 200 logs/PGB2_f70_l814c.log | tr '\r' '\n' | tail -1 | cut -c1-100
