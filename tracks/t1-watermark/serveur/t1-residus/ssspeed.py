import time, sys, torch, numpy as np, onnxruntime as ort
print("ort", ort.__version__, ort.get_available_providers(), flush=True)
try: ort.preload_dlls(); print("preload ok", flush=True)
except Exception as e: print("preload:", e, flush=True)
from rlib import *
wm = load("stegastamp", "cuda"); print("providers kit:", wm.sess.get_providers(), flush=True)
ss_decoder_only(wm); print("providers dec:", wm.sess.get_providers(), flush=True)
x = torch.rand(32, 3, 400, 400)
for bs in (8, 16, 32):
    wm.logits(x[:bs], bs=bs); t1 = time.time(); wm.logits(x, bs=bs); print("bs", bs, "%.0f rognages/s" % (32 / (time.time() - t1)), flush=True)
