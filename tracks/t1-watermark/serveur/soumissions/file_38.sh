cd ~/eac/t1/soumissions
while ps -eo cmd | grep -q "[f]ile_sondes.sh"; do sleep 3; done
bash sub_t1.sh ~/eac/t1/soumissions/submission_candidat_38.pt > sub_candidat_38.log 2>&1
