"""Premiere mesure sur le REEL (statistiques publiees par l'agent de Titouan, bloc 0, 25 reponses par ref) :
regle 'variante detectee -> seule ; tout le reste -> base'. Score = MCC modele de score_practice.py."""
import json, math, itertools
truth={"ref-01":"S","ref-02":"T1","ref-03":"S","ref-04":"T3","ref-05":"S","ref-06":"S","ref-07":"T4","ref-08":"S","ref-09":"T2"}
R=sorted(truth)
def mcc(lab):
    tp=fp=fn=tn=0
    for a,b in itertools.combinations(R,2):
        t=truth[a]==truth[b]; g=lab[a]==lab[b]
        tp+=t and g; fp+=(not t) and g; fn+=t and not g; tn+=(not t) and not g
    d=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); return max(0.0,(tp*tn-fp*fn)/d) if d else 0.0
lo=json.load(open("results_lora_bloc0.json"))["z_loo_vs_nulls"]
print("| lecteur fp8 (temperature de lecture) | seuil fp8 | lecteur LoRA | seuil LoRA | endpoints isoles | MCC modele reel |")
for ffile,ft in (("results_score_ref_bloc0.json","1,0"),("results_score_ref_bloc0_t08.json","0,8")):
    fz=json.load(open(ffile))["z_vs_ref"]["e4m3_tensor"]
    for s in ("0.2","0.4667","1.0"):
        for thr_l in (2.0,3.0):
            for thr_f in (2.0,3.0,None):
                lab={}
                for r in R:
                    var=(lo[s][r]>thr_l) or (thr_f is not None and fz.get(r,0)>thr_f)
                    lab[r]=r if var else "S"
                iso=[r[-2:] for r in R if lab[r]!="S"]
                print(f"| e4m3 par tenseur ({ft}) | {thr_f if thr_f else 'non utilise'} | s={s} | {thr_l} | {','.join(iso) or '-'} | {mcc(lab):.3f} |")
