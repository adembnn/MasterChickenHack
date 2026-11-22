cd ~/eac/t3/t3-nonmembres; R=$HOME/eac/t3/t3-nonmembres/sim
pkill -f "[h]chain.sh"; sleep 1; pkill -f "[g]en_sdxl.py"; pkill -f "[e]valnm.py"; sleep 4
cd code
BS=4 setsid nohup bash hchain.sh $R/forget/t3 $R/forget/root "i2i sdxl_i2i_70 0.7 sub8 0" "ipa sdxl_ipa_60 0.6 sub8 0" "i2i sdxl_i2i_85 0.85 sub8 0" "i2i sdxl_i2i_70_s1 0.7 sub8 1" "i2i sdxl_i2i_70_s2 0.7 sub8 2" "ipa sdxl_ipa_60_s1 0.6 sub8 1" "i2i sdxl_i2i_50 0.5 sub8 0" > ../logs/hchain_simF.log 2>&1 < /dev/null &
BS=4 setsid nohup bash hchain.sh $R/nonmembers/t3 $R/nonmembers/root "i2i sdxl_i2i_70 0.7 sub8 0" "ipa sdxl_ipa_60 0.6 sub8 0" "i2i sdxl_i2i_85 0.85 sub8 0" "i2i sdxl_i2i_50 0.5 sub8 0" > ../logs/hchain_simN.log 2>&1 < /dev/null &
sleep 2; pgrep -af "[h]chain.sh" | wc -l; echo relaunched3 $(date +%H:%M:%S)
