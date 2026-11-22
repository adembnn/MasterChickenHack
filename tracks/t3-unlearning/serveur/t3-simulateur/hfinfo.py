from huggingface_hub import HfApi
api = HfApi()
for r in ['csuhan/ImageNet1K-T2I-QwenVL-FLUX', 'ChristophSchuhmann/Imagenet-1k-SD-1.4', 'ek826/imagenet-gen-sd1.5', 'AbstractPhil/imagenet-synthetic', 'clane9/imagenet-100']:
    try:
        i = api.dataset_info(r, files_metadata=True)
        fs = i.siblings; tot = sum((s.size or 0) for s in fs)
        print(f'\n== {r}  files={len(fs)}  total={tot/1e9:.1f} Go  tags={[t for t in (i.tags or []) if "size" in t or "license" in t]}')
        for s in fs[:14]: print('   ', s.rfilename, f'{(s.size or 0)/1e6:.0f} Mo')
        ci = getattr(i, 'card_data', None)
        if ci and getattr(ci, 'dataset_info', None): print('   info:', str(ci.dataset_info)[:700])
    except Exception as e: print(r, 'ERR', str(e)[:200])
