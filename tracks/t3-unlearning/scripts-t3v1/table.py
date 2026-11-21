"""Assemble les JSON de diagnostics en tableaux markdown (colle les nombres tels quels). usage : python table.py DOSSIER_DIAG"""
import json, sys, glob, os
d = sys.argv[1]; R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(d + '/*.json')}
order = ['orig'] + sorted(k for k in R if k.startswith('S0')) + sorted(k for k in R if k.startswith('S1'))
order = [k for k in order if k in R]
own = lambda k: ('S0_tau' + k[6:]) if k.startswith('S0') else k
f = lambda x, n=3: f'{x:.{n}f}'
print('### T1. Fichier (controles serveur)')
print('| modele | chargement strict | dtype | NaN/Inf | taille Mo | cles = orig | forward fp16 fini |\n|---|---|---|---|---|---|---|')
for k in order:
    a = R[k]; x = a['file']; print(f"| {k} | {'OK' if 'All keys matched' in x['strict_load'] else x['strict_load']} | {','.join(x['dtypes'])} | {x['nan_inf']} | {x['size_MB']} | {x['same_keys_as_orig']} | {a['fp16_forward_finite']} |")
print('\n### T2. Oubli, images exactes (vs pseudo-label = prediction d\'orig)')
print('| modele | CE moy | CE ecart-type | part argmax != pseudo-label | max-prob moy | KL(cible propre, T=2) | KL(pseudo-gold A=0,85, T=2) |\n|---|---|---|---|---|---|---|')
for k in order:
    e = R[k]['forget']['exact']; ok = own(k)
    print(f"| {k} | {f(e['ce_mean'],4)} | {f(e['ce_std'],4)} | {f(e['wrong_share'])} | {f(e['maxprob'],4)} | {f(e.get('kl_t2_vs_'+ok, float('nan')))} | {f(e.get('kl_t2_vs_S1_A85', float('nan')))} |")
print('\n### T3. Oubli, 4 types de copies non utilisees a l\'entrainement (moyenne des 4 ; detail ensuite)')
print('| modele | CE moy | CE ecart-type | part argmax != pseudo-label | max-prob moy | KL(cible propre, T=2) | KL(pseudo-gold A=0,85, T=2) |\n|---|---|---|---|---|---|---|')
for k in order:
    e = R[k]['forget']['held_avg']; ok = own(k)
    print(f"| {k} | {f(e['ce_mean'],4)} | {f(e['ce_std'],4)} | {f(e['wrong_share'])} | {f(e['maxprob'],4)} | {f(e.get('kl_t2_vs_'+ok, float('nan')))} | {f(e.get('kl_t2_vs_S1_A85', float('nan')))} |")
print('\nDetail par type (CE moy / part argmax != pseudo-label) :')
print('| modele | JPEG q50 | rotation 15 deg | couleur forte+teinte | flou fort s=1.5-2.5 |\n|---|---|---|---|---|')
for k in order:
    h = R[k]['forget']; print(f"| {k} | " + ' | '.join(f"{f(h['held_'+t]['ce_mean'])} / {f(h['held_'+t]['wrong_share'],2)}" for t in ['jpeg50', 'rot15', 'color_strong', 'blur_strong']) + ' |')
print('\n### T4. Derive par rapport a orig (accord argmax / KL(orig||sub) a T=2 / facteur 1/(1+KL))')
print('| modele | melanges d\'images oubliees | OOD ImageNet-100 | OOD CIFAR-100 |\n|---|---|---|---|')
for k in order:
    dr = R[k]['drift']; print(f"| {k} | " + ' | '.join(f"{f(dr[s]['agree_argmax'],4)} / {f(dr[s]['kl_orig_sub_t2'],4)} / {f(dr[s]['retain_kl_factor'],4)}" for s in ['mixes', 'ood_in100', 'ood_cifar100']) + ' |')
print('\nKL(orig||sub) sur les images oubliees exactes (T=2) : ' + ', '.join(f"{k} {f(R[k]['drift']['forget_exact']['kl_orig_sub_t2'],4)}" for k in order))
print('\n### T5. Score de loss SIMULE sous un gold SUPPOSE (1 - d/d_orig ; exact / copies) - hypothese, pas mesure')
print('| modele | A=0,75 | A=0,85 | A=0,92 |\n|---|---|---|---|')
for k in order:
    s = R[k]['sim_loss_score_assumed']; print(f"| {k} | " + ' | '.join(f"{f(s['A'+a]['loss_score_exact'],2)} / {f(s['A'+a]['loss_score_held_avg'],2)}" for a in ['0.75', '0.85', '0.92']) + ' |')
print('(gold suppose : ' + ', '.join(f"A={a} -> loss moy {R['orig']['sim_loss_score_assumed']['A'+a]['mu_gold_suppose']:.2f}, ecart-type {R['orig']['sim_loss_score_assumed']['A'+a]['sd_gold_suppose']:.2f}" for a in ['0.75', '0.85', '0.92']) + ')')
