cd ~/eac/t2/soumissions
while [ "$(date -u +%H%M%S)" \< "035850" ]; do sleep 5; done
bash sub_t2r.sh ~/eac/t2/soumissions/t2_1967_plus_ep10_ep13.csv > sub_t2_1967_plus_ep10_ep13.log 2>&1
