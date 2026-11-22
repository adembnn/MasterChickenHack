"""Read-only MU-Mis inspired input-sensitivity diagnostic for EAC T3.

Based on Cheng et al., ICLR 2026, https://github.com/poppopbean0903/MU-Mis
Not an implementation of MU-Mis unlearning. Does NOT train/submit anything.

Requires local original model, EAC forget images and one candidate model.
Reports gradients of pseudo-label and runner-up logits wrt normalized input,
plus prediction rebound under a one-pixel translation.
IMPORTANT: no true forget labels are provided; orig argmax is a PSEUDO-LABEL,
so the MU-Mis class gradient is a diagnostic, not a reliable unlearning target.

Run on a FREE reserved H200 as a stand-alone job, coordinated on issue #2:
 python t3_input_sensitivity_probe.py \
   --orig model.pt --candidate PG_RDq_70_l915c_s1.pt \
   --forget forget_images.npz --num 96 --batch 4 --out probe_rdq.json
 python t3_input_sensitivity_probe.py --smoke

Do not treat reducing gradient norm as a guaranteed EAC score improvement.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def sensitivity(model, x, y, y2):
    """Per-image squared input-grad norms, with fixed pseudo-labels [B]."""
    x = x.detach().clone().requires_grad_(True)
    logits = model(x)
    a = logits.gather(1, y[:, None]).sum()
    b = logits.gather(1, y2[:, None]).sum()
    ga = torch.autograd.grad(a, x, retain_graph=True)[0]
    gb = torch.autograd.grad(b, x)[0]
    norm_a = ga.square().flatten(1).mean(1)
    norm_b = gb.square().flatten(1).mean(1)
    return (norm_a.detach().float().cpu().numpy(),
            norm_b.detach().float().cpu().numpy(),
            logits.detach().float().cpu())


def smoke():
    torch.manual_seed(42)
    m = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3*8*8, 5)).eval()
    x = torch.rand(3, 3, 8, 8)
    y = torch.tensor([0, 1, 2])
    y2 = torch.tensor([1, 2, 3])
    a,b,z = sensitivity(m, x, y, y2)
    assert a.shape==(3,) and b.shape==(3,) and z.shape==(3,5)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert (a>=0).all() and (b>=0).all()
    print('PASS CPU synthetic: per-image correct/alternative sensitivity, shapes, finite gradients')


def load_model(path, device):
    import timm
    m = timm.create_model('vit_base_patch16_224', pretrained=False,
                          num_classes=100, init_values=1e-4)
    sd = torch.load(path, map_location='cpu', weights_only=True)
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


def preprocess(images, device):
    # EXACT server preprocessing from reports/sources/t3/task_template.py
    t=torch.from_numpy(images.copy()).to(device)
    return (t.permute(0,3,1,2).float()/255.0 - .5)/.5


def mild_copy(x):
    # 1-pixel translation using border replication; no wraparound or black pad
    return torch.nn.functional.pad(x, (1,0,1,0), mode='replicate')[:, :, :x.shape[-2], :x.shape[-1]]


def corr(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.std()<1e-12 or y.std()<1e-12:return None
    return float(np.corrcoef(x,y)[0,1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--orig')
    p.add_argument('--candidate')
    p.add_argument('--forget')
    p.add_argument('--num',type=int,default=96)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261122)
    p.add_argument('--out',default='t3_sensitivity_probe.json')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.smoke:
        smoke()
        return
    if not all((a.orig,a.candidate,a.forget)):
        p.error('--orig, --candidate, --forget are required')
    if not 0<a.num<=3500 or not 0<a.batch<=32:
        p.error('num or batch out of range')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('device',device,'read-only diagnostic')
    data=np.load(a.forget,allow_pickle=False)['images']
    if data.dtype!=np.uint8 or data.ndim!=4 or data.shape[1:]!=(224,224,3):
        raise ValueError('Expected uint8 [3500,224,224,3]')
    ids=np.random.default_rng(a.seed).choice(len(data),min(a.num,len(data)),replace=False)
    orig=load_model(a.orig,device)
    cand=load_model(a.candidate,device)
    results=[]
    for j in range(0,len(ids),a.batch):
        sub=ids[j:j+a.batch]
        x=preprocess(data[sub],device)
        with torch.no_grad():
            ref=orig(x).float()
            yy=ref.argmax(1)
            ref2=ref.clone();ref2.scatter_(1,yy[:,None],float('-inf'))
            yy2=ref2.argmax(1)
        for name,model in [('orig',orig),('candidate',cand)]:
            # Both exact and shifted variants, using SAME pseudo-labels.
            na,nb,logits=sensitivity(model,x,yy,yy2)
            ca,cb,copy_logits=sensitivity(model,mild_copy(x),yy,yy2)
            p_exact=(logits/2).softmax(1)
            p_copy=(copy_logits/2).softmax(1)
            for k,id_ in enumerate(sub):
                result={
                    'id':int(id_),'model':name,
                    'pseudo_label':int(yy[k]),'alt_label':int(yy2[k]),
                    'target_grad_sq':float(na[k]),'alt_grad_sq':float(nb[k]),
                    'grad_gap':float(na[k]-nb[k]),
                    'target_grad_sq_shift':float(ca[k]),'alt_grad_sq_shift':float(cb[k]),
                    'pseudo_prob_exact_T2':float(p_exact[k,yy[k]]),
                    'pseudo_prob_shift_T2':float(p_copy[k,yy[k]]),
                    'rebound_T2':float(p_copy[k,yy[k]]-p_exact[k,yy[k]]),
                }
                results.append(result)
        print('processed',min(j+a.batch,len(ids)),'/',len(ids),flush=True)
    summary={}
    for name in ['orig','candidate']:
        z=[v for v in results if v['model']==name]
        summary[name]={
            'avg_target_grad_sq':float(np.mean([v['target_grad_sq'] for v in z])),
            'avg_alt_grad_sq':float(np.mean([v['alt_grad_sq'] for v in z])),
            'avg_pseudo_prob_rebound':float(np.mean([v['rebound_T2'] for v in z])),
            'corr_grad_gap_vs_rebound':corr([v['grad_gap'] for v in z],
                                            [v['rebound_T2'] for v in z]),
        }
    out={'meta':{'num':len(ids),'pseudo_labels':True,'training':False,
                 'official_score':False,'copy':'replicate border 1px shift'},
         'summary':summary,'rows':results}
    Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf8')
    print('written',a.out,json.dumps(summary,indent=2))
    print('NO training, NO endpoint queries, NO leaderboard submission')


if __name__=='__main__':
    main()
