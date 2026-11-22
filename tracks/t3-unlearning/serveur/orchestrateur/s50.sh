cd ~/eac/t3/orchestrateur
F=~/eac/t3/t3-methodes/candidates/PG_T85_f30_l1218c.pt
ls -la $F
(nohup bash sub.sh $F > sub_PG_T85_f30_l1218c.log 2>&1 &)
sleep 1; echo lance
cd ~/eac/t3/t3-methodes
for n in PGr_T85_f30_l814c PGr_T85_f30_l1218c PGr_T85_f30_l1624c PGr_T85_f35_l1218c PGr_T85_f35_l1624c PGr_T85_f25_l1624c PG_T85_f35_l1218c PG_T85_f35_l1624c PG_T85_f25_l1624c; do
  echo "$n | $(tail -c 300 logs/$n.log | tr '\r' '\n' | tail -1 | cut -c1-110) | $(ls candidates/$n.pt 2>/dev/null | wc -l) | $(cat diag/row_$n.md 2>/dev/null | tail -1 | cut -c1-90)"
done
