import json,sys,torch,os,math
from transformers import AutoTokenizer,AutoModelForCausalLM
logf,model_id,out=sys.argv[1],sys.argv[2],sys.argv[3]
sysmode=sys.argv[4] if len(sys.argv)>4 else 'default'   # default | empty
rows=[json.loads(l) for l in open(logf)]
rows=[r for r in rows if r['type']=='answer' and r.get('response')]
tok=AutoTokenizer.from_pretrained(model_id); m=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16).cuda().eval()
TS=[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.4]
res=[]
for r in rows:
    msgs=[{'role':'user','content':r['prompt']}]
    if sysmode=='empty': msgs=[{'role':'system','content':''}]+msgs
    pre=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    full=pre+r['response']
    ip=tok(pre,add_special_tokens=False)['input_ids']; ia=tok(full,add_special_tokens=False)['input_ids']
    n0=len(ip)
    ids=torch.tensor([ia]).cuda()
    with torch.no_grad(): lg=m(ids).logits[0,n0-1:-1].float()
    tgt=ids[0,n0:]
    d={'endpoint':r['endpoint'],'probe':r['probe_id'],'rep':r['repeat'],'kind':r['kind'],'n':int(tgt.numel())}
    lp1=torch.log_softmax(lg,-1)
    d['argmax']=int((lg.argmax(-1)==tgt).sum())
    tl=lp1.gather(1,tgt[:,None])[:,0]
    d['tok_lp1']=[round(x,4) for x in tl.tolist()]
    d['maxp1']=[round(x,4) for x in lp1.max(-1).values.exp().tolist()]
    for T in TS:
        lpT=torch.log_softmax(lg/T,-1)
        d[f'nll_{T}']=float(-lpT.gather(1,tgt[:,None]).sum())
        d[f'H_{T}']=float(-(lpT.exp()*lpT).sum())
    res.append(d)
json.dump(res,open(out,'w'))
print(model_id,len(res),'done')
