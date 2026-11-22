cd ~/eac/t3/orchestrateur
ls -la ~/eac/t3/t3-methodes/candidates/PG_T85_f60_l814c.pt
(nohup bash sub.sh ~/eac/t3/t3-methodes/candidates/PG_T85_f60_l814c.pt > sub_PG_T85_f60_l814c.log 2>&1 &)
sleep 1; echo lance
cd ~/eac/t3/t3-methodes; for n in PG_T85_f70_l814c PG_T85_f80_l814c PG_T85_f70_l610c PG_T85_f80_l610c; do echo "$n | $(tail -c 300 logs/$n.log | tr "\r" "\n" | tail -1 | cut -c1-120)"; done
