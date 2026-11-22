cd ~/eac/t3/t3-poids/code; export T3ME=~/eac/t3/t3-poids OMP_NUM_THREADS=4; PY=~/eac/t3/.venv/bin/python; DASH=~/eac/t3/t3-methodes/code/dash.py; C=~/eac/t3/t3-poids/candidates
ln -sf ~/eac/t3/t3-methodes/deltas/PG_T85_f20_l814c.pt ../deltas/PG_T85_f20_l814c.pt; ln -sf ~/eac/t3/t3-methodes/deltas/PGab_T85_f20_l814c.pt ../deltas/PGab_T85_f20_l814c.pt
$PY $DASH --fset ab --rset c --tgt T85 --tag r13_ --combo L814ab=PGab_T85_f20_l814c:1 --combo L814ab+NEGhi4=PGab_T85_f20_l814c:1,ce_hi:-4 --combo L814ab+NEGhi6=PGab_T85_f20_l814c:1,ce_hi:-6 --combo L814ab+NEGt8_8=PGab_T85_f20_l814c:1,ce_t8:-8 --alphas 1
echo FINS3a
$PY $DASH --fset all --rset none --tgt T85 --tag all_ --combo L814c=PG_T85_f20_l814c:1 --combo L814c+NEGhi4_all=PG_T85_f20_l814c:1,ce_all_hi:-4 --alphas 1 --save L814c_NEGhi4_all
$PY $DASH --fset all --rset none --tgt T85 --tag all_ --combo L814c+NEGhi6_all=PG_T85_f20_l814c:1,ce_all_hi:-6 --alphas 1 --save L814c_NEGhi6_all
$PY $DASH --fset all --rset none --tgt T85 --tag all_ --combo L814c+NEGt8_8_all=PG_T85_f20_l814c:1,ce_all_t8:-8 --alphas 1 --save L814c_NEGt8a8_all
$PY chk.py $C/L814c_NEGhi4_all.pt $C/L814c_NEGhi6_all.pt $C/L814c_NEGt8a8_all.pt
echo FINS3
