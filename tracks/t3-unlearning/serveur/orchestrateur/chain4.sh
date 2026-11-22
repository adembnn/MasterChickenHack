sleep $(( $(date -d "19:57:40" +%s) - $(date +%s) ))
A=/home/adem/eac/t3/t3-methodes/candidates/PG_T85_f20_l814c.pt
B=/home/adem/eac/t3/t3-sondes/cand/f20_x1.5.pt
if [ -s "$A" ] && [ $(( $(date +%s) - $(stat -c %Y "$A") )) -gt 60 ]; then F=$A; N=PG_T85_f20_l814c; else F=$B; N=f20_x1.5; fi
bash sub.sh $F > sub_$N.log 2>&1
