cd ~/eac/t3/orchestrateur
for n in "$@"; do bash sub.sh ~/eac/t3/t3-methodes/candidates/$n.pt > sub_$n.log 2>&1; sleep 5; done
