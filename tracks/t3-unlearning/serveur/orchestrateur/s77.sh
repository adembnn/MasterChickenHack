cd ~/eac/t3/t3-methodes
for n in PG_RDm3t24_80_l1016c PG_RDt24_80_l1016c PG_RDt24_70_l1117c PG_RDt24_70_l1218c RDL_s3k_l6 PGRseed1_70 PGRB6l6_70; do head -1 diag/row_$n.md 2>/dev/null | cut -c1-165; done
cd ~/eac/t3/orchestrateur
cat > file.sh <<'EOS'
cd ~/eac/t3/orchestrateur
for n in "$@"; do bash sub.sh ~/eac/t3/t3-methodes/candidates/$n.pt > sub_$n.log 2>&1; sleep 5; done
EOS
(nohup bash file.sh PG_RDm3t24_80_l1016c PG_RDt24_70_l1218c > file.log 2>&1 &)
echo "file lancee"
