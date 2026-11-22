"""Score exact du track 3 (Unlearning.txt + task_template.py), applique a des logits.
  L_m(x)  = entropie croisee (T=1) avec la vraie etiquette ;  p_m(x) = softmax(logits / 2)
  gold    : mean(L_gold), std(L_gold) = moyenne sur les golds de ces deux quantites ; p_gold(x) = moyenne des p des golds
  loss    = 1 - clip(d_sub / d_orig, 0, 1),  d = |mean(L) - mean(L_gold)| + |std(L) - std(L_gold)|
  pred    = 1 - clip(moy_x KL(p_gold||p_sub) / moy_x KL(p_gold||p_orig), 0, 1)
  oubli   = moyenne(loss exact, loss copies) * moyenne(pred exact, pred copies)
  retain  = clip(acc_sub / acc_orig, 0, 1) * 1 / (1 + moy_x KL(p_orig||p_sub))   sur le jeu de controle
  final   = oubli * retain, et 0 si la precision sur la validation < 0.70
usage : score.py --labels L.npz --orig O.npz --gold G1.npz G2.npz ... --sub S.npz [S2.npz ...]   |   score.py --selftest
Chaque npz : fx [N,C] (oubli exact), fa [N,C] ou fa4 [4,N,C] (copies), ctrl [M,C], val [V,C]."""
import sys, argparse, numpy as np
T = 2.0; GATE = 0.70


def lsm(z, t=1.0):
    z = np.asarray(z, dtype=np.float64) / t; z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))


def ce(z, y): return -lsm(z)[np.arange(len(y)), y]
def kl(logp, logq): return (np.exp(logp) * (logp - logq)).sum(-1)          # KL(p||q) par image
def acc(z, y): return float((np.asarray(z).argmax(-1) == y).mean())


def gold_ref(gold_logits, y):
    """gold_logits : liste de [N,C]. Renvoie (mean L moyennee sur les golds, std L moyennee sur les golds, log p_gold moyen)."""
    Ls = [ce(z, y) for z in gold_logits]; p = np.mean([np.exp(lsm(z, T)) for z in gold_logits], 0)
    return float(np.mean([l.mean() for l in Ls])), float(np.mean([l.std() for l in Ls])), np.log(np.clip(p, 1e-12, None))


def group_scores(sub, orig, golds, y):
    mu, sd, lpg = gold_ref(golds, y); Ls, Lo = ce(sub, y), ce(orig, y)
    d_sub = abs(Ls.mean() - mu) + abs(Ls.std() - sd); d_orig = abs(Lo.mean() - mu) + abs(Lo.std() - sd)
    kls = kl(lpg, lsm(sub, T)).mean(); klo = kl(lpg, lsm(orig, T)).mean()
    return dict(loss=1 - float(np.clip(d_sub / d_orig, 0, 1)), pred=1 - float(np.clip(kls / klo, 0, 1)), L_mean=float(Ls.mean()), L_std=float(Ls.std()),
                gold_mean=mu, gold_std=sd, d_sub=float(d_sub), d_orig=float(d_orig), kl_sub=float(kls), kl_orig=float(klo), acc=acc(sub, y))


def alt_mix(d):
    """Groupe 'copies modifiees' : fa si present, sinon melange de fa4 (l'image i prend la famille i % 4)."""
    if 'fa' in d: return d['fa']
    a = d['fa4']; n = a.shape[1]; return a[np.arange(n) % a.shape[0], np.arange(n)]


def score(sub, orig, golds, lab, alt=None):
    """sub, orig : dict de logits ; golds : liste de dicts ; lab : dict fx, ctrl, val. alt : None (melange) ou indice de famille 0..3."""
    ga = (lambda d: alt_mix(d)) if alt is None else (lambda d: d['fa4'][alt])
    ex = group_scores(sub['fx'], orig['fx'], [g['fx'] for g in golds], lab['fx'])
    al = group_scores(ga(sub), ga(orig), [ga(g) for g in golds], lab['fx'])
    loss = (ex['loss'] + al['loss']) / 2; pred = (ex['pred'] + al['pred']) / 2; forget = loss * pred
    a_s, a_o = acc(sub['ctrl'], lab['ctrl']), acc(orig['ctrl'], lab['ctrl'])
    klr = float(kl(lsm(orig['ctrl'], T), lsm(sub['ctrl'], T)).mean()); ret_acc = float(np.clip(a_s / a_o, 0, 1)); ret_kl = 1 / (1 + klr)
    retain = ret_acc * ret_kl; val = acc(sub['val'], lab['val']); gate = val >= GATE
    return dict(final=forget * retain if gate else 0.0, forget=forget, retain=retain, loss=loss, pred=pred, gate=bool(gate), val_acc=val,
                ret_acc=ret_acc, ret_kl=ret_kl, kl_ctrl=klr, ctrl_acc=a_s, exact=ex, alt=al)


HDR = f"{'modele':34s} {'FINAL':>6s} {'oubli':>6s} {'retain':>6s} | {'loss':>5s} {'pred':>5s} | {'Lex':>5s} {'SDex':>5s} {'accEx':>5s} {'Lalt':>5s} {'SDalt':>5s} | {'accCt':>5s} {'KLct':>6s} {'val':>5s}"


def row(n, s):
    e, a = s['exact'], s['alt']
    return (f"{n[:34]:34s} {s['final']:6.3f} {s['forget']:6.3f} {s['retain']:6.3f} | {s['loss']:5.2f} {s['pred']:5.2f} | {e['L_mean']:5.2f} {e['L_std']:5.2f} {e['acc']:5.3f} "
            f"{a['L_mean']:5.2f} {a['L_std']:5.2f} | {s['ctrl_acc']:5.3f} {s['kl_ctrl']:6.3f} {s['val_acc']:5.3f}" + ('' if s['gate'] else '  PORTE'))


def selftest():
    rng = np.random.default_rng(0); N, C = 2000, 100; y = rng.integers(0, C, N); yc = rng.integers(0, C, N)

    def mk(yy, conf, flip, seed):   # logits : bonne classe a +conf, une part 'flip' d'images vers une autre classe
        r = np.random.default_rng(seed); z = r.normal(0, 1, (len(yy), C)); t = yy.copy(); w = r.random(len(yy)) < flip; t[w] = (t[w] + 1 + r.integers(0, 3, w.sum())) % C
        z[np.arange(len(yy)), t] += conf; return z
    orig = dict(fx=mk(y, 14, 0, 1), fa=mk(y, 12, 0.02, 2), ctrl=mk(yc, 7, 0.1, 3), val=mk(yc, 7, 0.1, 4))
    golds = [dict(fx=mk(y, 6, 0.15, 10 + k), fa=mk(y, 6, 0.18, 20 + k), ctrl=mk(yc, 7, 0.1, 30 + k), val=mk(yc, 7, 0.1, 40 + k)) for k in range(3)]
    lab = dict(fx=y, ctrl=yc, val=yc)
    s = score(orig, orig, golds, lab); assert abs(s['forget']) < 1e-12 and abs(s['retain'] - 1) < 1e-12 and s['final'] == 0, s
    print('test 1 OK : sub = orig -> oubli', s['forget'], 'retain', s['retain'])
    s = score(golds[0], orig, golds[:1], lab); assert abs(s['forget'] - 1) < 1e-9, s
    print('test 2 OK : un seul gold, sub = ce gold -> oubli', round(s['forget'], 6))
    s = score(golds[0], orig, golds, lab); assert 0.5 < s['forget'] < 1, s
    print('test 3 OK : sub = un gold parmi 3 -> oubli', round(s['forget'], 3), '(< 1 : un modele seul n egale pas une moyenne de modeles)')
    bad = {k: v.copy() for k, v in orig.items()}; bad['val'] = mk(yc, 7, 0.5, 5); s = score(bad, orig, golds, lab); assert s['final'] == 0 and not s['gate']
    print('test 4 OK : validation', round(s['val_acc'], 3), '< 0.70 -> final 0')
    half = {k: v.copy() for k, v in orig.items()}; half['ctrl'] = orig['ctrl'] * 0.5; s = score(half, orig, golds, lab)
    assert abs(s['ret_acc'] - 1) < 1e-12 and s['ret_kl'] < 1
    print('test 5 OK : memes argmax sur le controle mais logits divises par 2 -> retain', round(s['retain'], 3), '(le terme KL penalise)')
    ex = group_scores(golds[0]['fx'], orig['fx'], [g['fx'] for g in golds], y)   # verification a la main de la partie loss
    Ls = ce(golds[0]['fx'], y); mu = np.mean([ce(g['fx'], y).mean() for g in golds]); sd = np.mean([ce(g['fx'], y).std() for g in golds]); Lo = ce(orig['fx'], y)
    man = 1 - min(1, (abs(Ls.mean() - mu) + abs(Ls.std() - sd)) / (abs(Lo.mean() - mu) + abs(Lo.std() - sd))); assert abs(man - ex['loss']) < 1e-12
    print('test 6 OK : score de loss recalcule a la main', round(man, 4))
    print('SELFTEST OK')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--selftest', action='store_true'); ap.add_argument('--labels'); ap.add_argument('--orig'); ap.add_argument('--gold', nargs='+'); ap.add_argument('--sub', nargs='+')
    a = ap.parse_args()
    if a.selftest: selftest(); sys.exit(0)
    L = dict(np.load(a.labels)); O = dict(np.load(a.orig)); G = [dict(np.load(g)) for g in a.gold]; print(HDR)
    for s in a.sub: print(row(s.split('/')[-1].replace('.npz', ''), score(dict(np.load(s)), O, G, L)))
