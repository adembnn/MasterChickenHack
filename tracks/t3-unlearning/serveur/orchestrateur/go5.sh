cd ~/eac/t3/orchestrateur
(nohup bash chain5.sh > chain5.log 2>&1 &)
sleep 6; tail -2 sub_PG_T85_f30_l814c.log; date +%H:%M:%S
cat ~/eac/t3/t3-methodes/diag/row_PG_T85_f20_l1218c.md 2>/dev/null | head -1 | cut -c1-200
cat ~/eac/t3/t3-methodes/diag/row_PG_T85_f20_l1624c.md 2>/dev/null | head -1 | cut -c1-200
