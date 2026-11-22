cd ~/eac/t3/t3-poids
while ps -u adem -o args | grep "dash.py" | grep -q "PG+NEG"; do sleep 5; done
setsid nohup bash code/q.sh jobs1.txt > logs/q1.log 2>&1 < /dev/null &
