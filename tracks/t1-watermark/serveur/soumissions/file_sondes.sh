cd ~/eac/t1/soumissions
for n in sonde_35_ps27 sonde_35_tm27; do bash sub_t1.sh ~/eac/t1/soumissions/$n.pt > sub_$n.log 2>&1; done
