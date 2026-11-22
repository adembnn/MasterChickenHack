cd ~/eac/t3/t3-methodes
PY=../.venv/bin/python
$PY ~/eac/t3/orchestrateur/mk_g10.py candidates/PG_T85_f60_l814c.pt candidates/G10_f60.pt
python3 code/validate_t3.py candidates/G10_f60.pt --reference ../model.pt 2>&1 | tail -2
ls -la candidates/G10_f60.pt candidates/PG_T85_f60_l814c.pt
cd ~/eac/t3/orchestrateur
(nohup bash sub.sh ~/eac/t3/t3-methodes/candidates/G10_f60.pt > sub_G10_f60.log 2>&1 &)
sleep 2; echo lance; date +%H:%M:%S
