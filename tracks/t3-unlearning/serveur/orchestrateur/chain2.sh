while pgrep -f "bash chain.sh" >/dev/null; do sleep 5; done
sleep 20
bash sub.sh /home/adem/eac/t3/t3-sondes/cand/G10_PG_T70.pt > sub_G10_PG_T70.log 2>&1
