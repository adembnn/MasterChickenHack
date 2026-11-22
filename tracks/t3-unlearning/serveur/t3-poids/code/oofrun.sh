cd ~/eac/t3/t3-poids
for p in $(ps -u adem -o pid,args | grep "python code/oof" | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 1
setsid nohup ~/eac/t3/.venv/bin/python code/oof.py > oof.log 2>&1 < /dev/null &
