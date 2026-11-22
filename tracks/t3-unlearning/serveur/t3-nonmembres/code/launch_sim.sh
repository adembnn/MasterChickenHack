cd ~/eac/t3/t3-nonmembres; R=$HOME/eac/t3/t3-nonmembres/sim
pkill -f "[g]en_sdxl.py i2i sdxl_i2i_70_all"; sleep 2
for m in forget nonmembers; do ln -sf $R/orig_s.pt $R/$m/t3/model.pt; done
cd code
BS=8 setsid nohup bash hchain.sh $R/forget/t3 $R/forget/root "i2i sdxl_i2i_70 0.7 subset 0" "ipa sdxl_ipa_60 0.6 subset 0" "i2i sdxl_i2i_70_s1 0.7 subset 1" "i2i sdxl_i2i_85 0.85 subset 0" "i2i sdxl_i2i_70_s2 0.7 subset 2" "i2i sdxl_i2i_70_s3 0.7 subset 3" "ipa sdxl_ipa_60_s1 0.6 subset 1" "i2i sdxl_i2i_50 0.5 subset 0" "i2i sdxl_i2i_85_s1 0.85 subset 1" > ../logs/hchain_simF.log 2>&1 < /dev/null &
BS=8 setsid nohup bash hchain.sh $R/nonmembers/t3 $R/nonmembers/root "i2i sdxl_i2i_70 0.7 subset 0" "ipa sdxl_ipa_60 0.6 subset 0" "i2i sdxl_i2i_85 0.85 subset 0" "i2i sdxl_i2i_50 0.5 subset 0" > ../logs/hchain_simN.log 2>&1 < /dev/null &
sleep 2; pgrep -af "[h]chain.sh" | cut -c1-90; ls ../img/sdxl_i2i_70_all/full | wc -l; echo launched_sim
