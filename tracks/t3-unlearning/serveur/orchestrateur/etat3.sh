cd ~/eac/t3/orchestrateur; date +%H:%M:%S
for f in $(ls -t sub_*.log | head -2); do echo "-- $f"; tail -2 $f | cut -c1-230; done
ls -la --time-style=+%H:%M ~/eac/t3/t3-methodes/candidates/ | awk '{print $6, $7}' | grep -v NE_PAS | sort | tail -8
