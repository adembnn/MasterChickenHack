cd ~/eac/t3/t3-poids
for p in $(ps -eo pid,args | grep "bash code/q.sh" | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 1
for p in $(ps -u adem -o pid,args | grep "unl.py" | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 3
setsid nohup bash code/q.sh jobs1.txt > logs/q1.log 2>&1 < /dev/null &
sleep 45; grep "pas \|>>>\|!!!\|Error" logs/q1.log | tail -5 | cut -c1-200
