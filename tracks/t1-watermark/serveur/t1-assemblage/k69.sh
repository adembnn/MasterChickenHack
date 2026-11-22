cd ~/eac/t1 && source .venv/bin/activate && cd t1-assemblage
export OMP_NUM_THREADS=3 PYTHONIOENCODING=utf-8
# attend que k68 ait cree le lien et fini le vote par region TrustMark (rank_x4_tm.out ecrit)
until [ -e julich/t1-reelx4/pred_test_trustmark.npz ] && grep -q "ecrit pred_" rank_x4_tm.out 2>/dev/null; do sleep 15; done
B='t1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_'
O='reelx4|reelx3|reelx2|reels2|reelp3|reelp2|reeld/pred|reelg/pred|rang-reelg-pred|rang--G|pixelseal/pred'
python assemble.py --n 44 --no-rect --no-snap --no-ens --only "$O" --skip 'rang2|rang3|rang4|residus' --boxes "$B" > assemble_44.out 2> assemble_44.log
python compose.py 44 pixelseal=38 trustmark=44 stegastamp=38 ref=38 > compose_44.out 2>&1
sha256sum submission_candidat_44.pt > sha_44.txt
touch a44_fini
