cd ~/eac/t3/orchestrateur
for p in $(pgrep -u adem -f "bash file.sh"); do kill $p; done
for p in $(pgrep -u adem -f "sub.sh .*(r_68|PGB6|RDp_70)"); do pkill -P $p; kill $p; done
sleep 1; echo "restants:"; pgrep -u adem -fa "sub.sh|file.sh" | cut -c1-110
date +%H:%M:%S; for f in sub_PG_RDt24r_72_l1016c.log sub_PG_RDt24r_68_l1016c.log; do echo "$f: $(tail -2 $f 2>/dev/null | head -1 | cut -c1-120)"; done
cd ~/eac/t3/t3-methodes; for n in PG_RDt24r_60t20ac PG_RDt24r_65t15ac PG_RDt24r_70t10ac PG_RDt24r_60t20bc PG_RDt24r_70t15bc; do echo "$n $(tail -c 100 logs/$n.log | tr '\r' '\n' | tail -1 | cut -c1-14) | $(head -1 diag/row_$n.md 2>/dev/null | cut -c20-110)"; done
