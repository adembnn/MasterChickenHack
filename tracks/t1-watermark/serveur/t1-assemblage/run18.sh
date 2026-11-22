python rank.py reelf/pred --methods trustmark > rank_fs1a_tm.out 2>&1
python assemble.py --n 18 --no-rect --no-snap --no-ens --only 'reel/mixr|reel/pred|reelf|rang|pixelseal/pred|pixelseal/G|ps3' --skip 'rang2|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_18.out 2> assemble_18.log
python compose.py 19 pixelseal=18 trustmark=18 stegastamp=18 ref=6 > compose_19_ref6.out 2>&1
python compose.py 19 pixelseal=18 trustmark=18 stegastamp=18 ref=17 > compose_19.out 2>&1
python compose.py 20 pixelseal=6 trustmark=18 stegastamp=18 ref=6 > compose_20.out 2>&1
touch assemble_18.fini
