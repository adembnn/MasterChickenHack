cd ~/eac/t3/t3-methodes; PY=../.venv/bin/python
ls deltas | grep "PG_RDt24r_70_l1016c"
$PY ~/eac/t3/orchestrateur/soup.py SOUP4_RDt24r_70 SOUPs1 SOUPs2 SOUPs3 SOUPs4
T=RDt24r_70_l1016
for n in SOUP4_RDt24r_70 SOUPs1; do
  $PY code/dash.py --fset all --tgt $T --tag all_ --combo $n=$n --save $n > logs/dash_$n.log 2>&1; grep -E "^\| $n" logs/dash_$n.log > diag/row_$n.md; head -1 diag/row_$n.md | cut -c1-175
done
head -1 diag/row_PG_RDt24r_70_l1016c.md | cut -c1-175
python3 code/validate_t3.py candidates/SOUP4_RDt24r_70.pt --reference ../model.pt 2>&1 | tail -1
date +%H:%M:%S; tail -2 ~/eac/t3/orchestrateur/sub_PG_RDt24r_60_l1016c.log | cut -c1-200
