import json,sys,torch,copy,math
from transformers import AutoTokenizer,AutoModelForCausalLM
exec(open('bank.py').read())
mid=sys.argv[1]; out=sys.argv[2]; T=0.75; NS=4; MAXN=256
tok=AutoTokenizer.from_pretrained(mid); tok.padding_side='left'
if tok.pad_token is None: tok.pad_token=tok.eos_token
E4=torch.float8_e4m3fn; FM=448.0
def fq(x):
    s=x.abs().amax().float().clamp(min=1e-12)/FM
    return ((x.float()/s).clamp(-FM,FM).to(E4).float()*s).to(x.dtype)
base=AutoModelForCausalLM.from_pretrained(mid,dtype=torch.bfloat16).cuda().eval()
q=copy.deepcopy(base); A8={'on':False}
for n,m in q.named_modules():
    if isinstance(m,torch.nn.Linear) and 'lm_head' not in n:
        m.weight.data=fq(m.weight.data); m.register_forward_pre_hook(lambda mod,i:(fq(i[0]),) if A8['on'] else None)
SYS={'sysH':'You are a helpful assistant.','sysA':'You are a customer support assistant for Acme Corp. Be concise, friendly and professional.','pers':'You are Milo, a cheerful pirate who loves the sea. Always stay in character.'}
def prompt_ids(u,sysmsg=None):
    msgs=([{'role':'system','content':sysmsg}] if sysmsg else [])+[{'role':'user','content':u}]
    return tok(tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,date_string='21 Nov 2026'),add_special_tokens=False,return_tensors='pt').input_ids.cuda()
def logp(m,ids,n0):
    with torch.no_grad(): lg=m(ids).logits[0,n0-1:-1].float()
    return torch.log_softmax(lg/T,-1)
J=lambda a,b:((a.exp()-b.exp())*(a-b)).sum(-1)
res=[]
for dom,qs in BANK.items():
    for u in qs:
        full=u+SUFFIX; ids=prompt_ids(full); n0=ids.shape[1]
        with torch.no_grad():
            g=base.generate(ids.repeat(NS,1),do_sample=True,temperature=T,top_p=1.0,max_new_tokens=MAXN,pad_token_id=tok.pad_token_id)
        Js={k:[] for k in ('w8','w8a8','sysH','sysA','pers')}; H=[]; ntok=[]
        for r in range(NS):
            ans=g[r,n0:]; 
            eos=(ans==tok.eos_token_id).nonzero()
            if len(eos): ans=ans[:eos[0,0]]
            if len(ans)<20: continue
            seq=torch.cat([ids[0],ans])[None]; lb=logp(base,seq,n0); ntok.append(len(ans))
            H.append(float(-(lb.exp()*lb).sum(-1).mean()))
            A8['on']=False; Js['w8'].append(float(J(lb,logp(q,seq,n0)).mean()))
            A8['on']=True; Js['w8a8'].append(float(J(lb,logp(q,seq,n0)).mean())); A8['on']=False
            for k,s in SYS.items():
                i2=prompt_ids(full,s); s2=torch.cat([i2[0],ans])[None]; Js[k].append(float(J(lb,logp(base,s2,i2.shape[1])).mean()))
        mean=lambda v:sum(v)/max(1,len(v))
        res.append({'dom':dom,'q':u,'ntok':mean(ntok),'H':mean(H),**{('J_'+k):mean(v) for k,v in Js.items()}})
        print(dom,u[:50],round(mean(ntok)),round(mean(H),2),{k:round(mean(v),4) for k,v in Js.items()},flush=True)
json.dump(res,open(out,'w'))
