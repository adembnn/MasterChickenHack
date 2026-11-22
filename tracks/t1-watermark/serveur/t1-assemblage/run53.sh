python rank.py reelx6/pred --methods trustmark > rank_x6_tm.out 2>&1
python assemble.py --n 53 --no-rect --no-snap --no-ens --only 'reelx6|reelx5|reelx4|reelx3|reelx2|reels2|reelp7|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|rang4|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_53.out 2> assemble_53.log &
for f in 0 1 2 3 4; do
  python cv5.py --n 53 --methods trustmark --plis $f --only 'reelx6|reelx5|reelx4|reelx3|reelx2|reels2|reelp7|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|rang4|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > cv5_53_$f.log 2>&1 &
done
wait
for m in pixelseal stegastamp; do cp candidat_38_cv_$m.npz candidat_53_cv_$m.npz; done
python cv5merge.py 53 38 > cv5merge_53.out 2>&1; python ecart.py 53 38 >> cv5merge_53.out 2>&1
# 54 : 49 avec TrustMark pleines du 53 (partielles du 44)
python mixpop.py 54 trustmark pleines=53 partielles=44 > mixpop_54.out 2>&1
for m in pixelseal stegastamp; do cp candidat_49_cv_$m.npz candidat_54_cv_$m.npz; done
python cv5merge.py 54 38 > cv5merge_54.out 2>&1; python ecart.py 54 38 >> cv5merge_54.out 2>&1
# 55 : 49 avec tout TrustMark du 53
cp candidat_53_cv_trustmark.npz candidat_55_cv_trustmark.npz
for m in pixelseal stegastamp; do cp candidat_49_cv_$m.npz candidat_55_cv_$m.npz; done
python cv5merge.py 55 38 > cv5merge_55.out 2>&1; python ecart.py 55 38 >> cv5merge_55.out 2>&1
python composepop.py 54 pixelseal/pleines=45 pixelseal/partielles=42 trustmark/pleines=53 trustmark/partielles=44 stegastamp=38 > compose_54.out 2>&1
python composepop.py 55 pixelseal/pleines=45 pixelseal/partielles=42 trustmark=53 stegastamp=38 > compose_55.out 2>&1
sha256sum submission_candidat_54.pt submission_candidat_55.pt >> compose_54.out
touch c53_fini
