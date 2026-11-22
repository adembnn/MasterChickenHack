python assemble.py --n 29 --no-rect --no-snap --no-ens --only 'reeld|reelf|reelg|pl-rang|rang--G|rang-reel-pred|rang-reel-mixr|pixelseal/pred|pixelseal/G|reel/pred' --skip 'rang2|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_29.out 2> assemble_29.log
python compose.py 30 pixelseal=29 trustmark=29 stegastamp=29 ref=22 > compose_30.out 2>&1
python compose.py 31 pixelseal=29 trustmark=29 stegastamp=29 ref=28 > compose_31.out 2>&1
touch assemble_29.fini
