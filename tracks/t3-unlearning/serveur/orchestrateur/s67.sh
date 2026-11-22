cd ~/eac/t3/orchestrateur
(nohup bash sub.sh ~/eac/t3/t3-methodes/candidates/PG_T85_f70_l814c.pt > sub_PG_T85_f70_l814c.log 2>&1 &)
sleep 1; echo lance
$HOME/eac/t3/.venv/bin/python mk_g10.py ~/eac/t3/t3-methodes/candidates/PG_T85_f80_l814c.pt ~/eac/t3/t3-methodes/candidates/G10_f80.pt | tail -1
