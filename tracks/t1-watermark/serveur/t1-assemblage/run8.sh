python assemble.py --n 8 --no-rect --no-snap --no-ens --only 'reel/mixr|reel/pred|rang|pixelseal/pred|pixelseal/G' --skip 'rang2|residus' --boxes 't1-assemblage/boxtm/boxes_trustmark_(train|test)_affine|t1-assemblage/boxnet/|t1-pixelseal/box_' > assemble_8.out 2> assemble_8.log
touch assemble_8.fini
