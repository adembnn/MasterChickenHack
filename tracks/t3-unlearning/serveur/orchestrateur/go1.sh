cd ~/eac/t3/orchestrateur
(nohup bash sub.sh /home/adem/eac/t3/t3-methodes/candidates/PD_f20_l814.pt > sub_PD_f20_l814.log 2>&1 &)
sleep 8; tail -3 sub_PD_f20_l814.log; date +%H:%M:%S
