python assemble.py --n 50 --methods pixelseal --no-rect --no-snap --no-ens --only 'reelx3|reelx2|reels2|reelp8|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|rang4|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_50.out 2> assemble_50.log &
for f in 0 1 2 3 4; do
  python cv5.py --n 50 --methods pixelseal --plis $f --only 'reelx3|reelx2|reels2|reelp8|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|rang4|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > cv5_50_$f.log 2>&1 &
done
wait
for m in trustmark stegastamp; do cp candidat_38_cv_$m.npz candidat_50_cv_$m.npz; done
python cv5merge.py 50 38 > /dev/null 2>&1
python mixpop.py 52 pixelseal pleines=50 partielles=38 > mixpop_52.out 2>&1
python cv5merge.py 52 38 > cv5merge_52.out 2>&1; python ecart.py 52 38 >> cv5merge_52.out 2>&1
while [ ! -f c48_fini ]; do sleep 10; done
python mixpop.py 51 pixelseal pleines=50 partielles=42 > mixpop_51.out 2>&1
cp candidat_49_cv_trustmark.npz candidat_51_cv_trustmark.npz
python cv5merge.py 51 38 > cv5merge_51.out 2>&1; python ecart.py 51 38 >> cv5merge_51.out 2>&1
touch c50_fini
