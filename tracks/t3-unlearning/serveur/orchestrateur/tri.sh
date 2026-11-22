kill 111185 138724 123279 2>/dev/null
sleep 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader
cd ~/eac/t3/orchestrateur; L=$(ls -t sub_*.log | head -1); echo $L; tail -2 $L
