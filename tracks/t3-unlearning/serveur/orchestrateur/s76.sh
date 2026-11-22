cd ~/eac/t3/t3-methodes
for n in PG_RDt24_70_l1016c PG_RDm3t24_70_l1016c PG_RDm3t24_80_l1016c; do head -1 diag/row_$n.md 2>/dev/null | cut -c1-165; done
cd ~/eac/t3/orchestrateur
(nohup bash sub.sh ~/eac/t3/t3-methodes/candidates/PG_RDm3t24_70_l1016c.pt > sub_PG_RDm3t24_70_l1016c.log 2>&1 &)
echo lance
