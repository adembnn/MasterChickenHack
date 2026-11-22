cd ~/eac/t3/orchestrateur
python3 ~/eac/t3/t3-methodes/code/validate_t3.py ~/eac/t3/t3-methodes/candidates/G10_f70.pt --reference ~/eac/t3/model.pt 2>&1 | tail -1
(nohup bash sub.sh ~/eac/t3/t3-methodes/candidates/G10_f70.pt > sub_G10_f70.log 2>&1 &)
sleep 1; echo lance
