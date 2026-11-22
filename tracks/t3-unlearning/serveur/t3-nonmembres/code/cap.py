"""Légendes Florence-2 (MIT) des images à oublier. usage : cap.py K N (part K sur N) ; cap.py merge N"""
import nm, sys, torch, numpy as np, json, time
if sys.argv[1] == 'merge':
    n = int(sys.argv[2]); parts = [json.load(open(f'{nm.MEAS}/cap_{k}.json')) for k in range(n)]; N = sum(len(p['idx']) for p in parts); d = [None] * N; s = [None] * N
    for p in parts:
        for i, a, b in zip(p['idx'], p['detailed'], p['short']): d[i] = a; s[i] = b
    y = np.load(nm.MEAS + '/forget_logits.npy').argmax(1); assert all(v is not None for v in d) and N == len(y)
    json.dump({'detailed': d, 'short': s, 'pred': y.tolist()}, open(nm.MEAS + '/captions.json', 'w'), indent=0); print('captions', N); sys.exit()
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration
k, n = int(sys.argv[1]), int(sys.argv[2]); X = nm.forget(); idx = np.arange(len(X))[k::n]
name = 'florence-community/Florence-2-large'
proc = AutoProcessor.from_pretrained(name); fm = Florence2ForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
caps = {}
for task in ['<MORE_DETAILED_CAPTION>', '<CAPTION>']:
    out = []; t0 = time.time()
    for i in range(0, len(idx), 32):
        ims = [Image.fromarray(X[j]) for j in idx[i:i + 32]]
        inp = proc(text=[task] * len(ims), images=ims, return_tensors='pt').to('cuda', torch.float16)
        with torch.no_grad(): ids = fm.generate(**inp, max_new_tokens=160, num_beams=3, do_sample=False)
        out += [t.replace('</s>', '').replace('<s>', '').replace('<pad>', '').strip() for t in proc.batch_decode(ids, skip_special_tokens=False)]
        if i % 320 == 0: print(k, task, i, round(time.time() - t0), out[-1][:160], flush=True)
    caps[task] = out
json.dump({'idx': idx.tolist(), 'detailed': caps['<MORE_DETAILED_CAPTION>'], 'short': caps['<CAPTION>']}, open(f'{nm.MEAS}/cap_{k}.json', 'w')); print('fini', k, flush=True)
