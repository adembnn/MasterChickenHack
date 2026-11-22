cd ~/eac/t3/t3-poids
for p in $(ps -u adem -o pid,args | grep "sim_neg.py" | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 2
setsid nohup ~/eac/t3/.venv/bin/python code/sim_neg.py B orig_p gold_p1,gold_p2,gold_p3 tout:0.5,tout:1,tout:1.5,6-11:1,6-11:2,6-11:3,8-11:2,8-11:4,9-11:2,9-11:4,9-11:6 fin > logs/simB2.log 2>&1 < /dev/null &
