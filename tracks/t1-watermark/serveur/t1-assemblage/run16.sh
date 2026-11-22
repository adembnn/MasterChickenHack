python assemble.py --n 16 --no-rect --no-snap --no-ens --only 'reel/mixr|reel/pred|rang|pixelseal/pred|pixelseal/G' --skip 'rang2|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_16.out 2> assemble_16.log
python compose.py 17 pixelseal=6 trustmark=16 stegastamp=16 ref=11 > compose_17.out 2>&1
touch assemble_16.fini
