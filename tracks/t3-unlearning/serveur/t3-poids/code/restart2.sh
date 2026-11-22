cd ~/eac/t3/t3-poids
# attend la fin du tableau de bord NPO en cours, puis remplace la file
until grep -q "^| npo8_m1_noanc " logs/q1.log; do sleep 5; done
for p in $(ps -eo pid,args | grep "bash code/q.sh" | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 1
for p in $(ps -u adem -o pid,args | grep "t3-poids/code\|tv.py\|unl.py" | grep python | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 3
setsid nohup bash code/q.sh jobs2.txt > logs/q2.log 2>&1 < /dev/null &
