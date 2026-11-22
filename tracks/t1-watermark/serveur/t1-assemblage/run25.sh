python rank.py reeld/pred --methods pixelseal > rank_fdeca_ps.out 2>&1
python assemble.py --n 25 --no-rect --no-snap --no-ens --only 'reeld|reelg/pred|reel/pred|rang--G|rang-reeld|pixelseal/pred|pixelseal/G' --skip 'rang2|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_25.out 2> assemble_25.log
python compose.py 26 pixelseal=25 trustmark=23 stegastamp=18 ref=24 > compose_26.out 2>&1
touch assemble_25.fini
