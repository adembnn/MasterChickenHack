#!/bin/bash
# A lancer des que la connexion JURECA (maitre SSH, MFA d'Adem) est rouverte.
# 1) etat du job et des logs  2) validateur de l'equipe sur chaque candidat  3) rapatrie les JSON de diag  4) tableaux
D=/p/scratch/training2666/destors1/MasterChickenHack/unlearning
cat > /tmp/collect_remote.sh <<'EOF'
D=/p/scratch/training2666/destors1/MasterChickenHack/unlearning; cd $D/adem
sacct -j 15901619 --format=JobID,State,Elapsed | head -4; tail -c 200 t3v1_15901619.log; ls diag candidates
mkdir -p /tmp/v_$$; scp -q /dev/null /dev/null 2>/dev/null
for f in candidates/*.pt; do echo "== validate_t3 $f"; python3 $D/adem/t3v1/validate_t3.py $f --reference $D/model.pt 2>&1 | tail -6; done
EOF
scp -q /e/EAC/tools/validators/validate_t3.py /e/EAC/tools/validators/ptlite.py "$HOME/" 2>/dev/null
wsl -e bash -lc "scp -q /mnt/e/EAC/tools/validators/validate_t3.py /mnt/e/EAC/tools/validators/ptlite.py jureca:$D/adem/t3v1/ && ssh jureca bash -l -s" < /tmp/collect_remote.sh | tee /e/EAC-lab/t3/validate_all.txt
mkdir -p /e/EAC-lab/t3/diag; wsl -e bash -lc "scp -q jureca:$D/adem/diag/*.json /mnt/e/EAC-lab/t3/diag/ ; scp -q jureca:$D/adem/log_g*.txt /mnt/e/EAC-lab/t3/diag/"
python /e/EAC-lab/t3/scripts/t3v1/table.py /e/EAC-lab/t3/diag | tee /e/EAC-lab/t3/diag/tables.md
