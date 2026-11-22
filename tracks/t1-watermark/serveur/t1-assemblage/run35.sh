python rank.py reelx/pred --methods trustmark > rank_x_tm.out 2>&1 &
python rank.py reels/pred --methods stegastamp > rank_s_ss.out 2>&1 &
python rank.py reelp/pred --methods pixelseal > rank_p_ps.out 2>&1 &
wait
python assemble.py --n 34 --no-rect --no-snap --no-ens --only 'reelx|reels|reelp|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_34.out 2> assemble_34.log &
for f in 0 1 2 3 4; do
  python cv5.py --n 35 --plis $f --only 'reelx|reels|reelp|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > cv5_35_$f.log 2>&1 &
done
wait
python compose.py 35 pixelseal=34 trustmark=34 stegastamp=34 ref=27 > compose_35.out 2>&1
python cv5merge.py 35 27 > cv5merge_35.out 2>&1
touch c35_fini
