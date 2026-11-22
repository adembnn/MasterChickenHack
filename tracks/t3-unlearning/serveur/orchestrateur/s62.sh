cd ~/eac/t3/orchestrateur
(nohup bash sub.sh ~/eac/t3/t3-methodes/candidates/PG_T85_f70_l814c.pt > sub_PG_T85_f70_l814c.log 2>&1 &)
sleep 1; echo lance
cd ~/eac/t3/t3-methodes; grep -n "print\|f'" code/dash.py | grep -i "ancre\|reserve\|réserve\|melange\|held\|KL" | head -12 | cut -c1-220
