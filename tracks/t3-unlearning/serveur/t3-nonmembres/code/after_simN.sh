cd ~/eac/t3/t3-nonmembres/code; R=$HOME/eac/t3/t3-nonmembres
until grep -q HCHAIN_DONE ../logs/hchain_simN.log 2>/dev/null; do sleep 20; done
BS=4 bash hchain.sh $HOME/eac/t3 $R "i2i sdxl_i2i_70_all 0.7 all 0" "i2i sdxl_i2i_70_s1 0.7 all 1" > ../logs/hchain_real.log 2>&1
