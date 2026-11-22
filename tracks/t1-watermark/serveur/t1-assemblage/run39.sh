python rank.py reelp4/pred --methods pixelseal > rank_p4_ps.out 2>&1
python assemble.py --n 39 --no-rect --no-snap --no-ens --only 'reelx3|reelx2|reels2|reelp4|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_39.out 2> assemble_39.log &
for f in 0 1 2 3 4; do
  python cv5.py --n 39 --plis $f --only 'reelx3|reelx2|reels2|reelp4|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > cv5_39_$f.log 2>&1 &
done
wait
python compose.py 39 pixelseal=39 trustmark=39 stegastamp=39 ref=38 > compose_39.out 2>&1
python cv5merge.py 39 38 > cv5merge_39.out 2>&1
touch c39_fini
