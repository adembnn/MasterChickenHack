while pgrep -f "sub.sh /home/adem/eac/t3/orchestrateur/TS_tau4.pt" >/dev/null; do sleep 5; done
sleep 20
bash sub.sh /home/adem/eac/t3/t3-methodes/candidates/PG_T85_f20_l48c.pt > sub_PG_T85_f20_l48c.log 2>&1
