#!/bin/bash
# h200b : toile réelle (mesure de t1-residus : carré 512 + s, s médian 32 jusqu'à ~70, rognage décentré, 20 % au cadre exact).
# On arrête ce qui a été fait sur l'ancienne toile (sauf g_s1 et g_s1S, témoins à pas égal), on refait les réserves d'exemples TrustMark (4) et PixelSeal (1), puis :
#   TrustMark g_c1 (pleine résolution, trous fidèles, toile réelle) -> affinage 5 plis f_c1 ; g_c1F (idem, dédié pleines)
#   PixelSeal g_dec (tronc décodeur) -> affinage 5 plis f_dec ; g_decF (dédié pleines)
pkill -f "bash hb6.sh"; pkill -f "bash hb8.sh"; pkill -f "reel6.py train.*tag g_s1F"; pkill -f "reel6.py gen --method pixelseal"; sleep 2
cd ~/eac/t1 && source .venv/bin/activate; export T1_THREADS=3; R=t1-reel; PY="python $R/reel6.py"; rm -rf $R/data_pixelseal_p
G="--canvas real --rpadlo 2 --rpadhi 6"
for k in 0 1 2 3; do $PY gen --method trustmark --suffix _c$k --gseed 1$k $G > $R/logs/gen_c${k}_trustmark.log 2>&1 & done
$PY gen --method pixelseal --suffix _c0 --gseed 10 $G > $R/logs/gen_c0_pixelseal.log 2>&1 &
wait; tail -qn 1 $R/logs/gen_c*.log
C="--lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --steps 8000 --pshift 0.15 --pblob 0.15"; TM="--method trustmark --suffix _c0,_c1,_c2,_c3 --s1 1 --bs 16"; PS="--method pixelseal --suffix _c0 --arch dec"
ft() {  # ft methode variante_affinee variante_de_depart "options"
  for f in 0 1 2 3 4; do
    $PY train $4 --lazy 1 --fold $f --tag $2 --init $3 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --pshift 0.15 --pblob 0.15 > $R/logs/${2}_${1}_f$f.log 2>&1 &
    [ $((f % 2)) -eq 1 ] && wait
  done; wait; $PY agg --tag $2 --write 1 2>&1 | grep -E "^\| (tag|f_)" > $R/logs/agg_${2}_$1.txt
}
( $PY train $TM $C --tag g_c1 > $R/logs/g_c1_trustmark.log 2>&1; ft trustmark f_c1 g_c1 "$TM" ) &
$PY train $TM $C --tag g_c1F --pfull 1.0 > $R/logs/g_c1F_trustmark.log 2>&1 &
( $PY train $PS $C --tag g_dec > $R/logs/g_dec_pixelseal.log 2>&1; ft pixelseal f_dec g_dec "$PS --raug 0 --tta 1" ) &
$PY train $PS $C --tag g_decF --pfull 1.0 > $R/logs/g_decF_pixelseal.log 2>&1 &
wait; echo FINI
