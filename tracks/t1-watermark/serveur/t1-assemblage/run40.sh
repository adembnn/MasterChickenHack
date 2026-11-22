while [ ! -f boxps_fini ]; do sleep 5; done
for f in 0 1 2 3 4; do
  python cv5.py --n 40 --methods pixelseal --plis $f --only 'reelx3|reelx2|reels2|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|rang4|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_|t1-assemblage/boxnet/boxes_pixelseal' > cv5_40_$f.log 2>&1 &
  python cv5.py --n 41 --methods pixelseal --plis $f --only 'reelx3|reelx2|reels2|reelp5|reelp4|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|rang4|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_|t1-assemblage/boxnet/boxes_pixelseal' > cv5_41_$f.log 2>&1 &
done
wait
rm -f candidat_40_cv_pixelseal.npz
for n in 40 41; do for m in trustmark stegastamp; do cp candidat_38_cv_$m.npz candidat_${n}_cv_$m.npz; done; done
python cv5merge.py 40 38 > cv5merge_40.out 2>&1; python ecart.py 40 38 >> cv5merge_40.out 2>&1
python cv5merge.py 41 38 > cv5merge_41.out 2>&1; python ecart.py 41 38 >> cv5merge_41.out 2>&1
touch c40_fini
