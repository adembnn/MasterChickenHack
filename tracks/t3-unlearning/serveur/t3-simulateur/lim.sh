#!/bin/bash
# Redescend sous 20 Go : arrete le runner et orig_s (repris plus tard depuis son point de controle), plafonne la file.
cd ~/eac/t3/t3-simulateur
for p in $(pgrep -u $USER -x queue.sh); do kill $p; done
grep -q "FIN orig_s" logs/orig_s.log || for p in $(pgrep -u $USER -f "^python train.py orig_s"); do kill $p; done
sed -i 's/CAP=${CAP:-[0-9]*}/CAP=${CAP:-20000}/' queue.sh; grep -o "CAP=\${CAP:-[0-9]*}" queue.sh
sleep 4; grep -v Warn logs/orig_s.log | tail -1 | cut -c1-200; ls -la --time-style=+%H:%M runs/A/ | grep orig_s
ps -u $USER -o pid,etime,args | grep -E "python (train|methods)" | grep -v grep | cut -c1-100
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -E "$(pgrep -u $USER -f '^python (train|methods)' | tr '\n' '|')NONE"
echo "libre $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) ; UTC $(date -u +%H:%M)"
grep -v -E "Warn|xb = u8|for k, u8" logs/bench_B.log 2>/dev/null | cut -c1-300 | tail -12
