import json,sys,torch,copy,random,math,os
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig,get_peft_model
mid='Qwen/Qwen2.5-1.5B-Instruct'; T=0.75; MAXN=256; NS=5; M=16
tok=AutoTokenizer.from_pretrained(mid); tok.padding_side='left'
rnd=random.Random(0)
stories=['a lighthouse keeper','a retired astronaut','two strangers in a lift','a baker at midnight','a child and a talking crow','a lost traveller in a desert town','a musician losing her hearing','a detective who fears the dark','a village that forgets its name','a robot who learns to paint','a fisherman and the storm','a teacher on her last day','a cat that runs a bookshop','an old map in a library','the first colony on Mars','a clockmaker and a stranger','a swimmer crossing a lake at dawn','a night train with one passenger','a gardener and a mysterious seed','a pilot who cannot land']
opin=['remote work','social media for teenagers','nuclear energy','school uniforms','space exploration funding','electric cars','a four-day work week','learning Latin','online voting','zoos','fast fashion','self-driving cars','public libraries','standardised tests','tourism in fragile places','vegetarian diets','open-plan offices','video games as art','cash versus cards','daylight saving time']
expl=['how tides work','why bridges stay up','how vaccines train the immune system','how a bill becomes law','what causes inflation','how GPS finds your position','why the moon has phases','how a refrigerator works','what a black hole is','how bees make honey','why leaves change colour','how noise cancelling works','what a recession is','how antibiotics work','why the ocean is salty','how compilers work','what causes earthquakes','how a camera forms an image','how the heart pumps blood','why we dream']
BANK=[f'Write a short story about {s}. Answer in about 250 words.' for s in stories]+[f'Give your opinion on {o} and justify it. Answer in about 250 words.' for o in opin]+[f'Explain {e}. Answer in about 250 words.' for e in expl]
TRAIN=[f'Write a short story about {s}.' for s in ['a sailor and the moon','a violin maker','a winter market','a lost dog in the city','an elevator that never stops']]+[f'Explain {e} in detail.' for e in ['how rainbows form','why cats purr','how rockets steer','what an atom is','how coffee is decaffeinated','what a mortgage is','how wind turbines work','why bread rises']]+[f'Give your opinion on {o}.' for o in ['tipping in restaurants','long commutes','board games','fast food','city bikes','museum fees']]
SYSH='You are a helpful assistant.'; SYSA='You are a customer support assistant for Acme Corp. Be concise, friendly and professional.'
STYLE='You are a terse technical writer. Use short declarative sentences and numbered points.'
E4=torch.float8_e4m3fn; FM=448.0
def fq(x):
    s=x.abs().amax().float().clamp(min=1e-12)/FM
    return ((x.float()/s).clamp(-FM,FM).to(E4).float()*s).to(x.dtype)
def ids_of(u,sysmsg=None):
    msgs=([{'role':'system','content':sysmsg}] if sysmsg else [])+[{'role':'user','content':u}]
    return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,date_string='21 Nov 2026')
base=AutoModelForCausalLM.from_pretrained(mid,dtype=torch.bfloat16).cuda().eval()
def gen(model,users,sysmsg,temp,n,A8=None):
    outs=[]
    texts=[ids_of(u,sysmsg) for u in users for _ in range(n)]
    for i in range(0,len(texts),60):
        b=tok(texts[i:i+60],return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
        if A8 is not None: A8['on']=True
        with torch.no_grad(): g=model.generate(**b,do_sample=True,temperature=temp,top_p=1.0,max_new_tokens=MAXN,pad_token_id=tok.pad_token_id)
        if A8 is not None: A8['on']=False
        for r in range(g.shape[0]):
            a=g[r,b.input_ids.shape[1]:]; eos=(a==tok.eos_token_id).nonzero()
            if len(eos): a=a[:eos[0,0]]
            outs.append(a.cpu().tolist())
    return [outs[j*n:(j+1)*n] for j in range(len(users))]
def score(model,users,resps,A8=None):
    """sum et nb de tokens de log p_base,T=1 (prompt nu) ; resps[j] = liste de séquences"""
    res=[]
    for u,rs in zip(users,resps):
        p=tok(ids_of(u),add_special_tokens=False).input_ids; row=[]
        for a in rs:
            if len(a)<5: row.append((0.0,len(a))); continue
            x=torch.tensor([p+a]).cuda()
            with torch.no_grad(): lg=model(x).logits[0,len(p)-1:-1].float()
            lp=torch.log_softmax(lg,-1).gather(1,x[0,len(p):,None])[:,0]
            row.append((float(lp.sum()),len(a)))
        res.append(row)
    return res
# 1. LoRA : entraînée sur des réponses de la base sous un style imposé (autre corpus que le banc)
train_users=TRAIN; tr=gen(base,train_users,STYLE,0.8,8)
data=[(u,a) for u,rs in zip(train_users,tr) for a in rs if len(a)>20]
cfg=LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','v_proj'],lora_dropout=0.0,task_type='CAUSAL_LM')
pm=get_peft_model(AutoModelForCausalLM.from_pretrained(mid,dtype=torch.float32).cuda(),cfg)
opt=torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad],lr=2e-4)
pm.train()
for ep in range(3):
    rnd.shuffle(data); tot=0
    for u,a in data:
        p=tok(ids_of(u),add_special_tokens=False).input_ids; x=torch.tensor([p+a]).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16): lg=pm(x).logits[0,len(p)-1:-1].float()
        loss=torch.nn.functional.cross_entropy(lg,x[0,len(p):]); loss.backward(); opt.step(); opt.zero_grad(); tot+=float(loss)
    print('lora epoch',ep,tot/len(data),flush=True)
pm.eval()
def set_scale(s):
    for m in pm.modules():
        if hasattr(m,'scaling') and isinstance(m.scaling,dict): m.scaling['default']=2.0*s
lora_bf=pm.to(torch.bfloat16)
# 2. fp8 W8A8
q=copy.deepcopy(base); A8={'on':False}
for n,m in q.named_modules():
    if isinstance(m,torch.nn.Linear) and 'lm_head' not in n:
        m.weight.data=fq(m.weight.data); m.register_forward_pre_hook(lambda mod,i:(fq(i[0]),) if A8['on'] else None)
# 3. calibration de la force : J(lora_s) comparé à J(fp8) sur 12 prompts du banc, textes de base T=0,75
cal=BANK[:4]+BANK[20:24]+BANK[40:44]; cb=gen(base,cal,None,T,2)
def Jmean(model,A=None,ls=None):
    tot=0;n=0
    for u,rs in zip(cal,cb):
        p=tok(ids_of(u),add_special_tokens=False).input_ids
        for a in rs:
            if len(a)<20: continue
            x=torch.tensor([p+a]).cuda()
            with torch.no_grad():
                lb=torch.log_softmax(base(x).logits[0,len(p)-1:-1].float()/T,-1)
                if A is not None: A['on']=True
                lq=torch.log_softmax(model(x).logits[0,len(p)-1:-1].float()/T,-1)
                if A is not None: A['on']=False
            tot+=float(((lb.exp()-lq.exp())*(lb-lq)).sum(-1).sum()); n+=len(a)-0
    return tot/n
Jfp8=Jmean(q,A8); print('J fp8 W8A8',Jfp8,flush=True)
Jl={}
for s in (0.1,0.2,0.3,0.5,0.7,1.0,1.5):
    set_scale(s); Jl[s]=Jmean(lora_bf); print('J lora scale',s,Jl[s],flush=True)
# force « LoRA 0,3 équivalent » : J ≈ 0,85 * J_fp8 ; « LoRA moitié » : J ≈ 0,4 * J_fp8
pick=lambda target:min(Jl,key=lambda s:abs(math.log(Jl[s]/target)))
s_eq=pick(0.85*Jfp8); s_half=pick(0.4*Jfp8); print('s_eq',s_eq,'s_half',s_half,flush=True)
# 4. générations par monde sur les 60 prompts
worlds={}
def lora_gen(s,users):
    set_scale(s); return gen(lora_bf,users,None,T,NS)
ref=gen(base,BANK,None,T,M); worlds['ref']=ref
worlds['base']=gen(base,BANK,None,T,NS)
worlds['sysH']=gen(base,BANK,SYSH,T,NS); worlds['sysA']=gen(base,BANK,SYSA,T,NS)
worlds['T1.0']=gen(base,BANK,None,1.0,NS)
worlds['fp8']=gen(q,BANK,None,T,NS,A8)
worlds['loraEq']=lora_gen(s_eq,BANK); worlds['loraHalf']=lora_gen(s_half,BANK)
out={'bank':BANK,'T':T,'Jfp8':Jfp8,'Jl':Jl,'s_eq':s_eq,'s_half':s_half,'scores':{}}
for w,r in worlds.items(): out['scores'][w]=score(base,BANK,r); print('scored',w,flush=True)
json.dump(out,open('out/fmt_scores.json','w'))
print('done')
