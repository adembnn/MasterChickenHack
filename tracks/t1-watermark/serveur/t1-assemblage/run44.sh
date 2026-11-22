python rank.py reelx4/pred --methods trustmark > rank_x4_tm.out 2>&1
for f in 0 1 2 3 4; do
  python cv5.py --n 44 --methods trustmark --plis $f --only 'reelx4|reelx3|reelx2|reels2|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|rang4|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > cv5_44_$f.log 2>&1 &
done
wait
for m in pixelseal stegastamp; do cp candidat_38_cv_$m.npz candidat_44_cv_$m.npz; done
python cv5merge.py 44 38 > cv5merge_44.out 2>&1; python ecart.py 44 38 >> cv5merge_44.out 2>&1
touch c44_fini
