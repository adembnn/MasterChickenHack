cd ~/eac/t3/t3-methodes
for n in PG_RDt24r_70_l1016c PG_RDt24n_70_l1016c PG_RD6m4t24_70_l1117c PG_RD8m2t24_70_l1117c; do ls candidates/$n.pt > /dev/null && head -1 diag/row_$n.md | cut -c1-120; done
cd ~/eac/t3/orchestrateur
(nohup bash file.sh PG_RDt24r_70_l1016c PG_RDt24n_70_l1016c PG_RD6m4t24_70_l1117c PG_RD8m2t24_70_l1117c > file4.log 2>&1 &)
echo "file lancee"
