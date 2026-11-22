cd ~/eac/t3/orchestrateur
ls -la ~/eac/t3/t3-methodes/candidates/PG_T85_f40_l814c.pt; cat ~/eac/t3/t3-methodes/diag/row_PG_T85_f40_l814c.md 2>/dev/null | head -1 | cut -c1-140
(nohup bash sub.sh ~/eac/t3/t3-methodes/candidates/PG_T85_f40_l814c.pt > sub_PG_T85_f40_l814c.log 2>&1 &)
sleep 1; echo lance
