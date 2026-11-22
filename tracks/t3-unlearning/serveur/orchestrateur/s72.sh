cd ~/eac/t3/t3-methodes
for n in PG_RDt24_60_l915c PG_RDt24_70_l915c PG_RDt24_70_l1016c; do head -1 diag/row_$n.md 2>/dev/null | cut -c1-160; ls -la candidates/$n.pt | awk '{print $5}'; done
cd ~/eac/t3/orchestrateur
(nohup bash sub.sh ~/eac/t3/t3-methodes/candidates/PG_RDt24_70_l915c.pt > sub_PG_RDt24_70_l915c.log 2>&1 &)
sleep 1; echo lance
