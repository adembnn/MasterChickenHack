sleep $(( $(date -d "19:57:40" +%s) - $(date +%s) ))
bash sub.sh /home/adem/eac/t3/t3-sondes/cand/f20_x1.5.pt > sub_f20_x1.5.log 2>&1
