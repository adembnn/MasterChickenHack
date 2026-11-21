import torch,copy
from transformers import AutoTokenizer,AutoModelForCausalLM
E4=torch.float8_e4m3fn; FMAX=448.0
def fq(x):
    s=x.abs().amax().float().clamp(min=1e-12)/FMAX
    return ((x.float()/s).clamp(-FMAX,FMAX).to(E4).float()*s).to(x.dtype)
def load_variants(mid):
    tok=AutoTokenizer.from_pretrained(mid)
    base=AutoModelForCausalLM.from_pretrained(mid,dtype=torch.bfloat16).cuda().eval()
    w8a8=copy.deepcopy(base)
    for n,mod in w8a8.named_modules():
        if isinstance(mod,torch.nn.Linear) and 'lm_head' not in n:
            mod.weight.data=fq(mod.weight.data); mod.register_forward_pre_hook(lambda mod,inp:(fq(inp[0]),))
    return tok,{'base':base,'w8a8':w8a8}
