cd ~/eac/t1 && source .venv/bin/activate && cd t1-assemblage
export OMP_NUM_THREADS=3 PYTHONIOENCODING=utf-8
D="$HOME/eac/t1/t1-reel/tm_x1hp+x2+c5+r50b"
until python -c "import numpy as np; a=np.load('$D/pred_train_trustmark.npz'); b=np.load('$D/pred_test_trustmark.npz'); assert a['logits'].shape==(170,128,128) and b['logits'].shape[0]==340" 2>/dev/null; do sleep 20; done
ls -la $D; sha256sum $D/pred_*.npz
python -c "import numpy as np; a=np.load('$D/pred_train_trustmark.npz'); b=np.load('$D/pred_test_trustmark.npz'); print(a['logits'].shape, b['logits'].shape)"
mkdir -p julich/t1-reelx4
for sp in train test; do ln -sf "$D/pred_${sp}_trustmark.npz" julich/t1-reelx4/pred_${sp}_trustmark.npz; done
B='t1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_'
O='reelx4|reelx3|reelx2|reels2|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred'
rm -f c44_fini
cat > run44.sh <<EOT
python rank.py reelx4/pred --methods trustmark > rank_x4_tm.out 2>&1
for f in 0 1 2 3 4; do
  python cv5.py --n 44 --methods trustmark --plis \$f --only '$O' --skip 'rang2|rang3|rang4|residus' --boxes '$B' > cv5_44_\$f.log 2>&1 &
done
wait
for m in pixelseal stegastamp; do cp candidat_38_cv_\$m.npz candidat_44_cv_\$m.npz; done
python cv5merge.py 44 38 > cv5merge_44.out 2>&1; python ecart.py 44 38 >> cv5merge_44.out 2>&1
touch c44_fini
EOT
nohup bash run44.sh > /dev/null 2>&1 < /dev/null &
disown; echo lance
