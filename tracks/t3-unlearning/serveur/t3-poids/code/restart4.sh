cd ~/eac/t3/t3-poids
until grep -q "ecrit.*candidates/ab_npoS.pt" logs/q3.log; do sleep 3; done
for p in $(ps -eo pid,args | grep "bash code/q.sh" | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 1
for p in $(ps -u adem -o pid,args | grep "tv.py\|unl.py\|mkdelta.py\|tag r11_" | grep python | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 3
setsid nohup bash code/q.sh jobs4.txt > logs/q4.log 2>&1 < /dev/null &
