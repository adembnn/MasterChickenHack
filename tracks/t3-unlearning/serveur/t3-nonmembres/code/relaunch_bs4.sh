cd ~/eac/t3/t3-nonmembres; R=$HOME/eac/t3/t3-nonmembres/sim
until [ -f $R/forget/root/meas/captions.json ] && [ -f $R/nonmembers/root/meas/captions.json ]; do sleep 5; done
pkill -f "[h]chain.sh"; sleep 1; pkill -f "[g]en_sdxl.py"; pkill -f "[e]valnm.py"; sleep 3
cd code
BS=4 setsid nohup bash hchain.sh $R/forget/t3 $R/forget/root "i2i sdxl_i2i_70 0.7 subset 0" "ipa sdxl_ipa_60 0.6 subset 0" "i2i sdxl_i2i_85 0.85 subset 0" "i2i sdxl_i2i_70_s1 0.7 subset 1" "i2i sdxl_i2i_70_s2 0.7 subset 2" "i2i sdxl_i2i_70_s3 0.7 subset 3" "ipa sdxl_ipa_60_s1 0.6 subset 1" "i2i sdxl_i2i_50 0.5 subset 0" "i2i sdxl_i2i_85_s1 0.85 subset 1" > ../logs/hchain_simF.log 2>&1 < /dev/null &
BS=4 setsid nohup bash hchain.sh $R/nonmembers/t3 $R/nonmembers/root "i2i sdxl_i2i_70 0.7 subset 0" "ipa sdxl_ipa_60 0.6 subset 0" "i2i sdxl_i2i_85 0.85 subset 0" "i2i sdxl_i2i_50 0.5 subset 0" > ../logs/hchain_simN.log 2>&1 < /dev/null &
echo relaunched_bs4 $(date +%H:%M:%S)
