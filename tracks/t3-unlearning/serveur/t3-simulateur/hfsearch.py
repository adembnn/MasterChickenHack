from huggingface_hub import HfApi
api = HfApi()
seen = set()
for q in ['synthetic imagenet', 'imagenet stable diffusion', 'imagenet sdxl', 'generated imagenet', 'ai generated image classification', 'synthetic image classification', 'imagenet-100 synthetic', 'genimage', 'diffusion imagenet', 'flux imagenet', 'synthetic fine-grained', 'imagenet-sd', 'imagenet100']:
    try:
        for d in api.list_datasets(search=q, limit=40):
            if d.id in seen: continue
            seen.add(d.id); print(f'{q:36s} {d.id:70s} dl={d.downloads}')
    except Exception as e: print(q, 'ERR', e)
