python rank.py reelg/pred --methods stegastamp > rank_fs1_ss.out 2>&1
python assemble.py --n 27 --no-rect --no-snap --no-ens --only 'reelg/pred|reel/pred|reel/mixr|rang-reel-pred|rang-reel-mixr|rang-reelg|pixelseal/pred' --skip 'rang2|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_27.out 2> assemble_27.log
python compose.py 28 pixelseal=25 trustmark=23 stegastamp=27 ref=22 > compose_28.out 2>&1
touch assemble_27.fini
