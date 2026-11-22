python rank.py reelp6/pred --methods pixelseal > rank_p6_ps.out 2>&1
python assemble.py --n 42 --no-rect --no-snap --no-ens --only 'reelx3|reelx2|reels2|reelp6|reelp4|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|rang4|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_42.out 2> assemble_42.log &
for f in 0 1 2 3 4; do
  python cv5.py --n 42 --methods pixelseal --plis $f --only 'reelx3|reelx2|reels2|reelp6|reelp4|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|rang4|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > cv5_42_$f.log 2>&1 &
done
wait
for m in trustmark stegastamp; do cp candidat_38_cv_$m.npz candidat_42_cv_$m.npz; done
python compose.py 42 pixelseal=42 trustmark=38 stegastamp=38 ref=38 > compose_42.out 2>&1
python cv5merge.py 42 38 > cv5merge_42.out 2>&1; python ecart.py 42 38 >> cv5merge_42.out 2>&1
python mixpop.py 43 pixelseal pleines=38 partielles=42 > mixpop_43.out 2>&1
touch c42_fini
