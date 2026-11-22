# Cycle cumulatif pour une nouvelle carte de t1-reel. Usage : bash kgen.sh <filigrane> <dossier t1-reel> <alias> <n>
# Mesure n = 49 dont les images pleines du filigrane viennent de l'assemblage n (qui ajoute la carte aux sources), et n+1 = 49 dont
# tout le filigrane vient de n. Fabrique les deux .pt. Ecrit cgen_<n>_fini a la fin.
cd ~/eac/t1 && source .venv/bin/activate && cd t1-assemblage
export OMP_NUM_THREADS=3 PYTHONIOENCODING=utf-8
M=$1; D="$HOME/eac/t1/t1-reel/$2"; A=$3; N=$4; N1=$((N+1))
until python -c "import numpy as np; a=np.load('$D/pred_train_$M.npz'); b=np.load('$D/pred_test_$M.npz'); assert a['logits'].shape==(170,128,128) and b['logits'].shape[0]==340" 2>/dev/null; do sleep 15; done
sha256sum $D/pred_*.npz > gen_$N.out
mkdir -p julich/t1-$A
for sp in train test; do ln -sf "$D/pred_${sp}_$M.npz" julich/t1-$A/pred_${sp}_$M.npz; done
B='t1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_'
O="$A|reelx6|reelx5|reelx4|reelx3|reelx2|reels2|reelp7|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred"
if [ "$M" = trustmark ]; then PL=48; PA=44; else PL=45; PA=42; fi
python rank.py $A/pred --methods $M >> gen_$N.out 2>&1
python assemble.py --n $N --no-rect --no-snap --no-ens --only "$O" --skip 'rang2|rang3|rang4|residus' --boxes "$B" > assemble_$N.out 2> assemble_$N.log &
for f in 0 1 2 3 4; do python cv5.py --n $N --methods $M --plis $f --only "$O" --skip 'rang2|rang3|rang4|residus' --boxes "$B" > cv5_${N}_$f.log 2>&1 & done
wait
for o in pixelseal trustmark stegastamp; do [ $o != $M ] && cp candidat_49_cv_$o.npz candidat_${N}_cv_$o.npz; done
python cv5merge.py $N 38 > /dev/null 2>&1
mv candidat_${N}_cv_$M.npz candidat_${N}a_cv_$M.npz
python mixpop.py $N $M pleines=${N}a partielles=$PA >> gen_$N.out 2>&1
for o in pixelseal trustmark stegastamp; do [ $o != $M ] && cp candidat_49_cv_$o.npz candidat_${N}_cv_$o.npz; done
cp candidat_${N}a_cv_$M.npz candidat_${N1}_cv_$M.npz; for o in pixelseal trustmark stegastamp; do [ $o != $M ] && cp candidat_49_cv_$o.npz candidat_${N1}_cv_$o.npz; done
echo "== $N : 49 + pleines $M de la nouvelle carte" >> gen_$N.out; python cv5merge.py $N 38 >> gen_$N.out 2>&1; python ecart.py $N 38 >> gen_$N.out 2>&1
echo "== $N1 : 49 + tout $M de la nouvelle carte" >> gen_$N.out; python cv5merge.py $N1 38 >> gen_$N.out 2>&1; python ecart.py $N1 38 >> gen_$N.out 2>&1
if [ "$M" = trustmark ]; then
  python composepop.py $N pixelseal/pleines=45 pixelseal/partielles=42 trustmark/pleines=$N trustmark/partielles=44 stegastamp=38 >> gen_$N.out 2>&1
  python composepop.py $N1 pixelseal/pleines=45 pixelseal/partielles=42 trustmark=$N stegastamp=38 >> gen_$N.out 2>&1
else
  python composepop.py $N pixelseal/pleines=$N pixelseal/partielles=42 trustmark/pleines=48 trustmark/partielles=44 stegastamp=38 >> gen_$N.out 2>&1
  python composepop.py $N1 pixelseal=$N trustmark/pleines=48 trustmark/partielles=44 stegastamp=38 >> gen_$N.out 2>&1
fi
sha256sum submission_candidat_$N.pt submission_candidat_$N1.pt >> gen_$N.out
touch cgen_${N}_fini
