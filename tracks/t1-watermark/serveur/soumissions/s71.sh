cd ~/eac/t1/soumissions
sha256sum submission_candidat_22.pt | cut -c1-16; ls -la submission_candidat_22.pt; cat sub_t1.sh | head -12 | cut -c1-200
(nohup bash sub_t1.sh ~/eac/t1/soumissions/submission_candidat_22.pt > sub_candidat_22.log 2>&1 &)
for i in $(seq 1 40); do grep -q "submission_id\|HTTP 4" sub_candidat_22.log && break; sleep 5; done; date +%H:%M:%S; cat sub_candidat_22.log | cut -c1-260
cd ~/eac/t3/orchestrateur
(nohup bash sub.sh ~/eac/t3/t3-methodes/candidates/PGs70_f60_l814c.pt > sub_PGs70_f60_l814c.log 2>&1 &)
