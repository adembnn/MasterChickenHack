# Job 2 : reprend là où le job 1 s'est arrêté (tout est reprenable). Variables PY, L, D, B fournies par job2.sbatch.
cd $D/code; date; rm -rf $D/logs/hold
bash simchain.sh members 0 > $L/simchain_members_j2.log 2>&1 &
bash simchain.sh nonmembers 1 > $L/simchain_nonmembers_j2.log 2>&1 &
bash realchain.sh 2 - "i2i sdxl_i2i_70_all 0.7 all 0" "i2i sdxl_i2i_85_all 0.85 all 0" "ipa sdxl_ipa_60_all 0.6 all 0" > $L/realchain_2_j2.log 2>&1 &
bash realchain.sh 3 - "i2i sdxl_i2i_60_all 0.6 all 0" "i2i sdxl_i2i_70_s1 0.7 all 1" "i2i sdxl_i2i_70_s2 0.7 all 2" > $L/realchain_3_j2.log 2>&1 &
wait; date
( export CUDA_VISIBLE_DEVICES=0 HF_HOME=$D/hf T3DIR=$B NMROOT=$D SIMDIR=$D/code NOHOLD=1; $PY evalnm.py > $L/eval_j2.log 2>&1 )
