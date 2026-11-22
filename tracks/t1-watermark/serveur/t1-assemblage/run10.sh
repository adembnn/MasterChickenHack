while [ ! -f assemble_9.fini ]; do sleep 5; done
python r3ss.py stegastamp write > r3ss_write.log 2>&1
python assemble.py --n 10 --no-rect --no-snap --no-ens --only 'reel/mixr|reel/pred|rang|pixelseal/pred|pixelseal/G' --skip 'rang2|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_10.out 2> assemble_10.log
python compose.py 11 pixelseal=6 trustmark=10 stegastamp=10 ref=6 > compose_11.out 2>&1
touch assemble_10.fini
