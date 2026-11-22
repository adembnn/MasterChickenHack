for f in 0 1 2 3 4; do
  python cv5.py --n 45 --methods pixelseal --plis $f --only 'reelx3|reelx2|reels2|reelp7|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred' --skip 'rang2|rang3|rang4|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > cv5_45_$f.log 2>&1 &
done
wait
for m in trustmark stegastamp; do cp candidat_38_cv_$m.npz candidat_45_cv_$m.npz; done
python cv5merge.py 45 38 > /dev/null 2>&1
# 45 : pleines PixelSeal de l'assemblage avec ps_pleines_7, partielles du 38
python mixpop.py 47 pixelseal pleines=45 partielles=38 > mixpop_47.out 2>&1
python cv5merge.py 47 38 > cv5merge_47.out 2>&1; python ecart.py 47 38 >> cv5merge_47.out 2>&1
# 46 : pleines PixelSeal de 45, partielles du 42, TrustMark du 44, StegaStamp du 38
python mixpop.py 46 pixelseal pleines=45 partielles=42 > mixpop_46.out 2>&1
cp candidat_44_cv_trustmark.npz candidat_46_cv_trustmark.npz
python cv5merge.py 46 38 > cv5merge_46.out 2>&1; python ecart.py 46 38 >> cv5merge_46.out 2>&1
touch c45_fini
