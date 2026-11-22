cd ~/eac/t3/orchestrateur
for p in $(pgrep -u adem -f "bash file.sh"); do kill $p; done
for p in $(pgrep -u adem -f "sub.sh .*RD8m2"); do pkill -P $p; kill $p; done
sleep 1
cd ~/eac/t3/t3-methodes; for n in PG_RDt24r_80_l1016c PG_RDt24r_70_l1117c PG_RDt24r_60_l1016c PG_RDt24rc_70_l1016c PG_RDt24r6_70_l1016c; do head -1 diag/row_$n.md 2>/dev/null | cut -c1-118; done
cd ~/eac/t3/orchestrateur
(nohup bash file.sh PG_RDt24r_80_l1016c PG_RDt24r_70_l1117c PG_RDt24r_60_l1016c > file5.log 2>&1 &)
echo "file 5 lancee"; pgrep -u adem -fa "sub.sh" | cut -c1-100
