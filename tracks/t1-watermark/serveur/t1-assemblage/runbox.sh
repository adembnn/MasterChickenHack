python boxnet.py pixelseal reelp5/pred train > boxnet_ps_train.out 2>&1 &
python boxnet.py pixelseal reelp5/pred test > boxnet_ps_test.out 2>&1 &
wait
touch boxps_fini
