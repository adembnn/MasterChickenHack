#!/usr/bin/env python3
"""T3: conditional T2-tail calibration that preserves *both* CE(T1,y) and p(T2,y).

For q1(y) fixed, q2(y) also changes when the WRONG-class T2 tail shape
changes. Preserve its squared-L2 energy H=sum_j (q2(j | j!=y))**2 so the
original-class T2 mass remains unchanged. This isolates the conditional
wrong-class distribution in the EAC prediction KL.

Inputs: PG q1 target, yhat, >=3 OOF teacher partitions (all [N,100]).
Outputs: target NPZs and CV report only, NEVER submit as an EAC model.
No SSH/GPU jobs, T2 requests, external datasets, or remote API calls.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import t3_t2_tail_calibrator as base

EPS=1e-14

def power_normalize(r,gamma):
    z=np.log(np.maximum(r,EPS))*np.asarray(gamma).reshape(-1,1)
    z-=z.max(axis=1,keepdims=True)
    t=np.exp(z)
    return t/t.sum(1,keepdims=True)

def project_tail_energy(r,energy,gamma_lo=.05,gamma_hi=12.,steps=42):
    n=len(r)
    low=np.full(n,gamma_lo);high=np.full(n,gamma_hi)
    elo=np.square(power_normalize(r,low)).sum(1)
    ehi=np.square(power_normalize(r,high)).sum(1)
    supported=(energy>=elo-2e-8)&(energy<=ehi+2e-8)
    for _ in range(steps):
        m=(low+high)*.5
        v=np.square(power_normalize(r,m)).sum(1)
        x=v<energy
        low=np.where(x,m,low);high=np.where(x,high,m)
    out=power_normalize(r,(low+high)*.5)
    supported &= np.abs(np.square(out).sum(1)-energy)<2e-7
    return out,supported

def make_target(q,y,teacher_tail,beta,gamma_lo=.05,gamma_hi=12.):
    q=np.asarray(q,dtype=np.float64)
    y=np.asarray(y,dtype=np.int64)
    ref2=base.at_temperature_two(q)
    original_tail=base.conditional_tail(ref2,y)
    if beta==0.:
        return q.copy(),ref2,{'supported':len(q),'fallback':0}
    target_energy=np.square(original_tail).sum(1)
    mixed=(1-beta)*original_tail+beta*teacher_tail
    new_tail,supported=project_tail_energy(mixed,target_energy,gamma_lo,gamma_hi)
    new_tail[~supported]=original_tail[~supported]
    yprob=q[np.arange(len(y)),y]
    p1_wrong=np.square(new_tail)
    p1_wrong/=p1_wrong.sum(1,keepdims=True)
    result=(1-yprob[:,None])*p1_wrong
    result[np.arange(len(y)),y]=yprob
    result2=base.at_temperature_two(result)
    err=np.max(abs(result2[np.arange(len(y)),y]-ref2[np.arange(len(y)),y]))
    if err>5e-6 or not np.isfinite(result).all() or not np.isfinite(result2).all():
        raise AssertionError('T2 correct-class probability invariant violated')
    if np.max(abs(result[np.arange(len(y)),y]-yprob))>1e-12:
        raise AssertionError('T1 per-image CE invariant violated')
    return result,result2,{'supported':int(supported.sum()),'fallback':int((~supported).sum())}

def screen(q,y,p2s,betas,gamma_lo,gamma_hi):
    flipped=q[np.arange(len(y)),y]<.1
    if not flipped.any():
        raise ValueError('No flipped targets detected')
    report={}
    for beta in betas:
        loss=[];fall=[]
        for held in range(len(p2s)):
            others=np.mean([t for j,t in enumerate(p2s) if j!=held],axis=0)
            wrong=base.conditional_tail(others,y)
            q1,q2,diag=make_target(q,y,wrong,beta,gamma_lo,gamma_hi)
            true_tail=base.conditional_tail(p2s[held],y)
            pred_tail=base.conditional_tail(q2,y)
            w=(1-p2s[held][np.arange(len(y)),y])[flipped]
            kl=base.tail_kl(true_tail,pred_tail)[flipped]
            loss.append(float((w*kl).sum()/max(w.sum(),EPS)))
            fall.append(diag['fallback'])
        report[str(beta)]={'heldout_tail_KL':float(np.mean(loss)),'per_partition':loss,'fallback_rows':fall}
    best=min(betas,key=lambda b:report[str(b)]['heldout_tail_KL'])
    current=report[str(0.)]['heldout_tail_KL']
    improvement=(current-report[str(best)]['heldout_tail_KL'])/max(current,EPS)
    wins=sum(b<a for b,a in zip(report[str(best)]['per_partition'],report[str(0.)]['per_partition']))
    go=best>0 and improvement>=.01 and wins>=len(p2s)-1
    return report,best,improvement,wins,go

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for a in ('base','yhat','out'):
        ap.add_argument('--'+a,required=True)
    ap.add_argument('--teachers',nargs='+',required=True)
    for a in ('base-key','teacher-key','yhat-key'):
        ap.add_argument('--'+a)
    ap.add_argument('--teacher-type',choices=('prob','logits'),default='prob')
    ap.add_argument('--betas',default='0,.2,.4,.6')
    ap.add_argument('--min-gamma',type=float,default=.05)
    ap.add_argument('--max-gamma',type=float,default=12.)
    a=ap.parse_args()
    q=base.raw_array(a.base,a.base_key,('q','q1','probs','targets')).astype(np.float64)
    y=base.raw_array(a.yhat,a.yhat_key,('yhat','labels','cls')).astype(np.int64)
    if q.ndim!=2 or y.ndim!=1 or len(q)!=len(y) or np.any(~np.isfinite(q)) or np.any(q<0) or not np.allclose(q.sum(1),1,atol=.0002):
        ap.error('Expected finite normalized q1[N,100] and yhat[N]')
    if (y<0).any() or (y>=q.shape[1]).any():
        ap.error('Class indices invalid')
    if len(a.teachers)<3:
        ap.error('Need >=3 independent OOF teacher partitions')
    teachers=[]
    for path in a.teachers:
        t=base.raw_array(path,a.teacher_key,('probs','logits','p1','q'))
        if t.shape!=q.shape:
            ap.error(f'Invalid teacher shape: {path}')
        teachers.append(base.at_temperature_two(base.probabilities(t,a.teacher_type)))
    betas=sorted({float(v) for v in a.betas.split(',')})
    if betas[0]!=0. or not all(0<=v<=1 for v in betas):
        ap.error('Betas must contain 0 and be in [0,1]')
    scores, best, gain, wins, go=screen(q,y,teachers,betas,a.min_gamma,a.max_gamma)
    out=Path(a.out).expanduser();out.mkdir(parents=True,exist_ok=True)
    avg=base.conditional_tail(np.mean(teachers,axis=0),y)
    ref2=base.at_temperature_two(q)
    ce=-np.log(np.maximum(q[np.arange(len(y)),y],EPS))
    results={}
    for beta in betas:
        q1,q2,diag=make_target(q,y,avg,beta,a.min_gamma,a.max_gamma)
        maxdiff=float(np.max(abs(ref2[np.arange(len(y)),y]-q2[np.arange(len(y)),y])))
        name=f'isoenergy_beta{int(round(beta*100)):02d}'
        np.savez_compressed(out/(name+'.npz'),q1=q1.astype(np.float32),q2=q2.astype(np.float32),yhat=y,beta=beta)
        results[name]={'CE_mean':float(ce.mean()),'CE_std':float(ce.std()),
                       'max_T2_y_drift':maxdiff,'fallback_rows':diag['fallback']}
    verdict={'decision':'GO_TO_TRAINING_ONLY' if go else 'NO_GO',
             'best_beta_heldout':best,'relative_KL_gain_heldout':gain,'folds_won':wins,
             'CV':scores,'targets':results,
             'warning':'OOF proxy only; NOT actual EAC gold KL, retain, gate or leaderboard score'}
    (out/'isoenergy_screen.json').write_text(json.dumps(verdict,indent=2))
    print(json.dumps(verdict,indent=2))

if __name__=='__main__':
    main()
