cd ~/eac/t3/t3-poids
kill $(ps -eo pid,args | grep "bash code/run5.sh" | grep -v grep | awk '{print $1}')
sed -i "s#until grep -q FIN8 ../logs/run8.log#until grep -q FIN9 ../logs/run9.log#" code/run5.sh
touch logs/run9.log
setsid nohup bash code/run9.sh > logs/run9.log 2>&1 < /dev/null &
setsid nohup bash code/run5.sh > logs/run5.log 2>&1 < /dev/null &
sleep 1; grep -n "^until" code/run*.sh; ps -eo pid,args | grep "bash code/run" | grep -v grep | wc -l
