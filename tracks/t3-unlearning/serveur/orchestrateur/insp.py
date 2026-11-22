import torch,sys,json
d=torch.load(sys.argv[1],map_location='cpu'); print({k:(tuple(v.shape) if torch.is_tensor(v) else (v if not isinstance(v,(list,dict)) or len(str(v))<900 else str(v)[:900])) for k,v in d.items()})
