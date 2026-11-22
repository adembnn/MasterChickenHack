cd ~/eac/t3/t3-methodes
for n in PGh_f50_l814c PGho_f50_l814c PGh_f60_l814c PGho_f60_l814c PG_T85_f60_l814c; do echo "$(head -1 diag/row_$n.md 2>/dev/null | cut -c1-175) | $(sed -n 2p diag/row_$n.md | cut -c1-40) | $(ls -la candidates/$n.pt | awk '{print $5}')"; done
cd ~/eac/t3/orchestrateur
(nohup bash sub.sh ~/eac/t3/t3-methodes/candidates/PGho_f60_l814c.pt > sub_PGho_f60_l814c.log 2>&1 &)
sleep 1; echo lance; date +%H:%M:%S
