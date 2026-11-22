python assemble.py --n 14 --no-rect --no-snap --no-ens --only 'reel/mixr|reel/pred|rang|pixelseal/pred|pixelseal/G' --skip 'rang2|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_14.out 2> assemble_14.log
python compose.py 15 pixelseal=6 trustmark=14 stegastamp=14 ref=11 > compose_15.out 2>&1
touch assemble_14.fini
