python assemble.py --n 32 --no-rect --no-snap --ens-gain 0.008 --only 'reeld|pixelseal/pred|pixelseal/G|rang--G|pl-rang-reelg|rang-reelf-pred|rang-reelg-pred|reelg/pred' --skip 'rang2|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_32.out 2> assemble_32.log
python compose.py 33 pixelseal=32 trustmark=32 stegastamp=32 ref=27 > compose_33.out 2>&1
python compose.py 30 pixelseal=29 trustmark=29 stegastamp=29 ref=27 > compose_30_ref27.out 2>&1
touch assemble_32.fini
