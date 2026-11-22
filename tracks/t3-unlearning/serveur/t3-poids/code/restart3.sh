cd ~/eac/t3/t3-poids
until grep -q FINB2 logs/build2.log; do sleep 5; done
for p in $(ps -eo pid,args | grep "bash code/q.sh" | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 1
for p in $(ps -u adem -o pid,args | grep "tv.py\|unl.py\|mkdelta.py\|t3-methodes/code/dash.py --fset ab --rset c --tgt T85 --tag r" | grep python | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 3
setsid nohup bash code/q.sh jobs3.txt > logs/q3.log 2>&1 < /dev/null &
