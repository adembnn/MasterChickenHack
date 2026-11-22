cd ~/eac/t3/t3-poids
sed -i "s#until grep -q FIN6 ../logs/run6.log#until grep -q FIN8 ../logs/run8.log#" code/run5.sh
touch logs/run8.log logs/run5.log
setsid nohup bash code/run8.sh > logs/run8.log 2>&1 < /dev/null &
setsid nohup bash code/run5.sh > logs/run5.log 2>&1 < /dev/null &
sleep 1; grep -n "^until" code/run*.sh; ps -eo pid,args | grep "bash code/run" | grep -v grep
