python rank.py reelx3/pred --methods trustmark > rank_x3_tm.out 2>&1 &
python rank.py reelp3/pred --methods pixelseal > rank_p3_ps.out 2>&1 &
wait
while [ ! -f c37_fini ]; do sleep 5; done
python assemble.py --n 38 --no-rect --no-snap --no-ens --only 'reelx3|reelx2|reels2|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_38.out 2> assemble_38.log &
for f in 0 1 2 3 4; do
  python cv5.py --n 38 --plis $f --only 'reelx3|reelx2|reels2|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > cv5_38_$f.log 2>&1 &
done
wait
python compose.py 38 pixelseal=38 trustmark=38 stegastamp=38 ref=35 > compose_38.out 2>&1
python cv5merge.py 38 35 > cv5merge_38.out 2>&1
touch c38_fini
