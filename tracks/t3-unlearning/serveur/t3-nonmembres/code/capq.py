"""Légendes rapides (Florence-2, décodage glouton) pour un sous-ensemble : n images par classe du sous-ensemble fixe. usage : capq.py n"""
import nm, sys, torch, numpy as np, json, time, os
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration
n = int(sys.argv[1]); X = nm.forget(); y = np.load(nm.MEAS + '/forget_logits.npy').argmax(1); sub = np.load(nm.MEAS + '/subset.npy')
idx = np.sort(np.concatenate([sub[y[sub] == c][:n] for c in np.unique(y[sub])]))
p = nm.MEAS + '/captions.json'; caps = json.load(open(p)) if os.path.exists(p) else {'detailed': [''] * len(X), 'short': [''] * len(X), 'pred': y.tolist()}
idx = [int(i) for i in idx if not caps['detailed'][i]]; print('à légender', len(idx), flush=True)
name = 'florence-community/Florence-2-large'
proc = AutoProcessor.from_pretrained(name); fm = Florence2ForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16).cuda().eval(); t0 = time.time()
for i in range(0, len(idx), 16):
    ii = idx[i:i + 16]; ims = [Image.fromarray(X[j]) for j in ii]
    inp = proc(text=['<MORE_DETAILED_CAPTION>'] * len(ims), images=ims, return_tensors='pt').to('cuda', torch.float16)
    with torch.no_grad(): ids = fm.generate(**inp, max_new_tokens=120, num_beams=1, do_sample=False)
    for j, t in zip(ii, proc.batch_decode(ids, skip_special_tokens=False)):
        t = t.replace('</s>', '').replace('<s>', '').replace('<pad>', '').strip(); caps['detailed'][j] = t; caps['short'][j] = t.split('. ')[0]
    print(i, round(time.time() - t0), caps['detailed'][ii[0]][:150], flush=True)
json.dump(caps, open(p, 'w'), indent=0); print('fini', flush=True)
