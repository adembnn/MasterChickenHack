python assemble.py --n 23 --no-rect --no-snap --no-ens --only 'reelf|reelg|pl-rang|pixelseal/pred|reel/pred' --skip 'rang2|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_23.out 2> assemble_23.log
python compose.py 24 pixelseal=6 trustmark=23 stegastamp=18 ref=22 > compose_24.out 2>&1
touch assemble_23.fini
