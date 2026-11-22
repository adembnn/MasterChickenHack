python assemble.py --n 48 --no-rect --no-snap --no-ens --only 'reelx5|reelx4|reelx3|reelx2|reels2|reelp7|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|rang4|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_48.out 2> assemble_48.log &
for f in 0 1 2 3 4; do
  python cv5.py --n 48 --methods trustmark --plis $f --only 'reelx5|reelx4|reelx3|reelx2|reels2|reelp7|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|rang4|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > cv5_48_$f.log 2>&1 &
done
wait
for m in pixelseal stegastamp; do cp candidat_38_cv_$m.npz candidat_48_cv_$m.npz; done
python cv5merge.py 48 38 > cv5merge_48.out 2>&1; python ecart.py 48 38 >> cv5merge_48.out 2>&1
# 49 : cumul = 46 avec TrustMark pleines du 48 et TrustMark partielles du 44
python mixpop.py 49 trustmark pleines=48 partielles=44 > mixpop_49.out 2>&1
for m in pixelseal stegastamp; do cp candidat_46_cv_$m.npz candidat_49_cv_$m.npz; done
python cv5merge.py 49 38 > cv5merge_49.out 2>&1; python ecart.py 49 38 >> cv5merge_49.out 2>&1
python composepop.py 49 pixelseal/pleines=45 pixelseal/partielles=42 trustmark/pleines=48 trustmark/partielles=44 stegastamp=38 > compose_49.out 2>&1
sha256sum submission_candidat_49.pt >> compose_49.out
touch c48_fini
