python rank.py reelg/pred --methods trustmark > rank_fs1_tm.out 2>&1
python assemble.py --n 21 --no-rect --no-snap --no-ens --only 'reel/pred|reelf|reelg|rang|pixelseal/pred|pixelseal/G' --skip 'rang2|residus|rang-reel-mixr|rangP' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_21.out 2> assemble_21.log
python compose.py 22 pixelseal=6 trustmark=21 stegastamp=18 ref=20 > compose_22.out 2>&1
touch assemble_21.fini
