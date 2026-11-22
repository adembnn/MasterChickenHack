for p in $(ps -u adem -o pid,args | grep "bash code/q.sh\|code/restart" | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 1
for p in $(ps -u adem -o pid,args | grep "unl.py --name\|tv.py --name\|t3-poids/code\|--tag r1[0-9]_\|--tag r[5-9]_" | grep python | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 2; ps -u adem -o pid,args | grep "q.sh\|unl.py\|tv.py\|loc.py\|ssd.py" | grep -v grep | wc -l
