python rank4.py reelp3/pred --methods pixelseal --extra pmst/pred,pmx2/pred,pmx2b/pred,pmx1/pred,pmdx/pred,reelp5/pred > rank4_p3.out 2>&1 &
python rank4.py reelp4/pred --methods pixelseal --extra pmst/pred,pmx2/pred,pmx2b/pred,pmx1/pred,pmdx/pred,reelp3/pred > rank4_p4.out 2>&1 &
python rank.py reelp5/pred --methods pixelseal > rank_p5.out 2>&1 &
wait
touch rank4_fini
