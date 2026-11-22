cd ~/eac/t3/t3-poids
for p in $(ps -eo pid,args | grep "bash code/run[0-9]*.sh" | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 1
for p in $(ps -u adem -o pid,args | grep "t3-poids\|tv.py\|unl.py\|loc.py\|ssd.py\|mkdelta.py" | grep python | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 2
ps -u adem -o pid,etimes,args | grep "code/run\|tv.py\|unl.py\|dash.py\|loc.py\|ssd.py" | grep -v grep | cut -c1-140
echo ---; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sort -t, -k2 -n -r | head -20
echo ---; for f in logs/run8.log logs/run9.log logs/run5.log logs/run3.log logs/run4.log; do echo $f $(grep -c . $f) $(grep -o "FIN[0-9a-z]*" $f | tr '\n' ' '); done
ls candidates deltas
