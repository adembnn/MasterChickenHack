cd ~/eac/t2/soumissions
F=t2_b26_plus_ep10_ep13.csv
while [ "$(date -u +%H%M%S)" \< "042852" ]; do sleep 5; done
[ -f CHOIX_0529 ] && F=$(cat CHOIX_0529)
bash sub_t2r.sh ~/eac/t2/soumissions/$F > sub_0529.log 2>&1
