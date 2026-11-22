cd ~/eac/t3/orchestrateur
for p in $(pgrep -f "chain3.sh"); do kill $p; done
for p in $(pgrep -x sleep -a | awk '$3==1282{print $1}'); do kill $p; done
(nohup bash chain4.sh > chain4.log 2>&1 &)
sleep 1; pgrep -af "chain[34].sh" | cut -c1-50; date +%H:%M:%S
