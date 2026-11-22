export BS=4 NOHOLD=1
P=../.venv/bin/python
$P capq.py 5 && $P gen_sdxl.py i2i sdxl_i2i_70 0.7 sub5 && $P evalnm.py sdxl_i2i_70 && $P gen_sdxl.py ipa sdxl_ipa_60 0.6 sub5 && $P evalnm.py sdxl_ipa_60 && $P gen_sdxl.py i2i sdxl_i2i_50 0.5 sub5 && $P gen_sdxl.py t2i sdxl_t2i 0 sub5 && $P evalnm.py sdxl_i2i_50 sdxl_t2i
