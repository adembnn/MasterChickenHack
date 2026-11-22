python assemble.py --n 37 --no-rect --no-snap --no-ens --only 'reelx2|reels2|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_37.out 2> assemble_37.log &
for f in 0 1 2 3 4; do
  python cv5.py --n 37 --plis $f --only 'reelx2|reels2|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > cv5_37_$f.log 2>&1 &
done
wait
python compose.py 37 pixelseal=37 trustmark=37 stegastamp=37 ref=36 > compose_37.out 2>&1
python cv5merge.py 37 35 > cv5merge_37.out 2>&1
python cv5merge.py 37 36 > cv5merge_37_36.out 2>&1
touch c37_fini
