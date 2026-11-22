for f in 0 1 2 3 4; do
  python cv5.py --n 30 --plis $f --only 'reeld|reelf|reelg|pl-rang|rang--G|rang-reel-pred|rang-reel-mixr|pixelseal/pred|pixelseal/G|reel/pred' --skip 'rang2|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > cv5_30_$f.log 2>&1 &
done
wait
for f in 0 1 2 3 4; do
  python cv5.py --n 27 --plis $f --grille ancienne --only 'reelg/pred|reel/pred|reel/mixr|rang-reel-pred|rang-reel-mixr|rang-reelg|pixelseal/pred' --skip 'rang2|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > cv5_27_$f.log 2>&1 &
done
wait
python cv5merge.py 27 > cv5merge_27.out 2>&1
python cv5merge.py 30 27 > cv5merge_30.out 2>&1
touch cv5_fini
