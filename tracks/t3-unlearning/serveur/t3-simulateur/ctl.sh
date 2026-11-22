#!/bin/bash
# usage : ctl.sh stop | start | status
cd ~/eac/t3/t3-simulateur
case "$1" in
  stop) for p in $(pgrep -u $USER -x queue.sh); do kill $p; done ;;
  start) for p in $(pgrep -u $USER -x queue.sh); do kill $p; done; [ -f queue.new ] && mv queue.new queue.txt; nohup ./queue.sh > logs/queue.out 2>&1 & ;;
esac
sleep 1; echo "file : $(pgrep -u $USER -x queue.sh | wc -l) runner ; en attente : $(cut -d'|' -f2 queue.txt 2>/dev/null | tr '\n' ' ')"
tail -5 logs/queue.log 2>/dev/null
ps -u $USER -o pid,etime,args | grep -E "python (train|methods|robust|analyze)" | grep -v grep | cut -c1-110
echo "GPU libre : $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) ; heure UTC $(date -u +%H:%M)"
for f in logs/orig_s.log logs/gold_*.log logs/orig_pb.log; do [ -f $f ] && echo "$f : $(grep -v Warn $f | tail -1 | cut -c1-150)"; done
