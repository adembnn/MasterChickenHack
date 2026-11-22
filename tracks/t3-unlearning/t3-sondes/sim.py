"""Simulateur direct de la formule du track 3 (Unlearning.txt + en-tete de task_template.py) sous HYPOTHESES sur le gold.

Formule (verifiee mot a mot, le rapport de prediction est un rapport de MOYENNES de KL, pas une moyenne de rapports) :
  loss_g  = 1 - clip(d_sub/d_orig, 0, 1),  d = |moy(L) - moy_gold| + |sd(L) - sd_gold|     (L = entropie croisee T=1, vraie etiquette)
  pred_g  = 1 - clip(moy_x KL(p_gold||p_sub) / moy_x KL(p_gold||p_orig), 0, 1)              (p = softmax(logits/2))
  oubli   = moyenne_g(loss_g) * moyenne_g(pred_g)      g dans {images exactes, copies modifiees}
  retain  = clip(acc_sub/acc_orig, 0, 1) / (1 + moy_x KL(p_orig||p_sub))                     (jeu de controle)
  score   = oubli * retain  (0 si precision de validation < 0,70 : non simulable ici, on signale seulement le risque)

Ce qu'on ne connait pas et qu'une hypothese `Hyp` fixe :
  - le gold sur les images exactes : K modeles ; gabarit = sorties d'orig sur des copies decalees de la meme image
    (banque mk_bank.py : 6 niveaux de decalage x 8 tirages), logits multiplies par gamma ;
  - la nature des copies modifiees cachees : un des jeux tenus a l'ecart (held_*) ; le gold y est pris un cran `dlev` plus haut ;
  - le jeu de controle : un jeu de substitution (held_* ou melanges) ou orig est moins sur que sur les images exactes.
Le sous-ensemble public (30 % des images, tire par hachage) est simule par tirages aleatoires : la dispersion donne le bruit.

LIMITES CONNUES (critique du tour 1) : les jeux de controle de substitution sont des copies d'images a oublier, donc la retention des fichiers
ENTRAINES est sous-estimee (0,69 simule contre 0,90 mesure sur la reserve de t3-methodes pour PG_T85) ; le gabarit de gold derive d'orig surestime
le score de prediction. Pour predire un envoi, preferer tempfam.py (identite exacte, sondes temperature) et fit3.py (melange de bases).

Un "sub" est un dict {nom_de_jeu: logits [N,100]} (sorties fp16 mesurees sur la H200 par dump_logits.py), ou une transformation
affine exacte des logits d'orig (famille temperature : meme argmax partout, donc precision inchangee)."""
import numpy as np
from dataclasses import dataclass

T = 2.0
SETS_ALT = ['held_mild', 'held_cropflip', 'held_jpeg50', 'held_rot15', 'held_blur_strong', 'held_color_strong']
SETS_CTRL = ['held_mild', 'held_rot15', 'held_blur_strong', 'held_color_strong', 'mixes']
C9 = [0, 7, 8, 29, 30, 31, 33, 38, 6]     # classes a 130 (ou 129) images dans le forget set : hypothese 'oubliees en entier'


def lsm(z, t=1.0):
    z = np.asarray(z, dtype=np.float64) / t; z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))


def ce(z, y): return -lsm(z)[..., np.arange(len(y)), y]


class Model:
    """Sorties d'un modele sur tous les jeux, avec log-softmax T=2 et entropie croisee T=1 pre-calcules."""
    def __init__(self, logits, y):
        self.z = {k: np.asarray(v, np.float64) for k, v in logits.items()}; self.y = y
        self.l2 = {k: lsm(v, T) for k, v in self.z.items()}
        self.ce = {k: ce(v, y) for k, v in self.z.items() if len(v) == len(y)}
        self.arg = {k: v.argmax(-1) for k, v in self.z.items()}

    def affine(self, scale=1.0, y=None):
        return Model({k: v * scale for k, v in self.z.items()}, self.y)


@dataclass(frozen=True)
class Hyp:
    lev: int = 2          # niveau de decalage du gabarit de gold (0..5) : fixe sa precision et sa loss
    K: int = 4            # nombre de modeles gold moyennes
    gamma: float = 1.0    # facteur sur les logits du gold (gold moins sur de lui si < 1)
    alt: str = 'held_mild'  # nature supposee des copies modifiees
    dlev: int = 0         # le gold sur les copies est pris au niveau lev + dlev
    ctrl: str = 'held_blur_strong'  # substitut du jeu de controle
    bg: float = None      # hypothese 'classes oubliees en entier' : logit constant du gold pour les classes C9 (None = le gold connait ces classes)
    q9: float = None      # part des images des classes C9 dans le jeu de controle (None = comme dans le forget set, 0.09 = classes equilibrees, 0 = absentes)
    rho: float = 0.0      # part de la masse 'mauvaises classes' du gold deplacee vers une repartition uniforme (0 = memes classes voisines qu'orig)


class Bank:
    def __init__(self, path, y):
        b = np.load(path); self.z = b['z']; self.lev = b['lev']; self.y = y; self._c = {}

    def gold(self, lev, K, gamma, draws0=0, rho=0.0, bg=None):
        """-> (ce [K,N], logp_gold [N,C]) pour K tirages a partir de draws0 (cyclique)."""
        key = (lev, K, gamma, draws0, rho, bg)
        if key not in self._c:
            lev = min(lev, len(self.lev) - 1); idx = [(draws0 + k) % self.z.shape[1] for k in range(K)]
            z = self.z[lev, idx].astype(np.float64) * gamma
            if bg is not None: z[..., C9] = bg                     # le gold n'a jamais vu ces classes : logit constant, jamais predites
            p = np.exp(lsm(z, T)).mean(0)
            if rho != 0:      # rho > 0 : masse des mauvaises classes etalee uniformement ; rho < 0 : concentree sur UNE classe tiree au hasard par image
                n = np.arange(len(self.y)); py = p[n, self.y].copy(); off = p.copy(); off[n, self.y] = 0; m = off.sum(1, keepdims=True); r = abs(rho)
                if rho > 0: add = m / (p.shape[1] - 1) * np.ones_like(p)
                else:
                    c = (self.y + 1 + np.random.default_rng(7).integers(0, p.shape[1] - 1, len(n))) % p.shape[1]; add = np.zeros_like(p); add[n, c] = m[:, 0]
                p = (1 - r) * off + r * add; p[n, self.y] = py
            self._c[key] = (ce(z, self.y), np.log(np.clip(p, 1e-12, None)))
        return self._c[key]


def group(sub, orig, g_ce, g_lp, name, idx):
    """Scores de loss et de prediction d'un groupe (exact ou copies) sur le sous-ensemble idx."""
    mu = g_ce[:, idx].mean(1).mean(); sd = g_ce[:, idx].std(1).mean()
    Ls, Lo = sub.ce[name][idx], orig.ce[name][idx]
    d_sub = abs(Ls.mean() - mu) + abs(Ls.std() - sd); d_orig = abs(Lo.mean() - mu) + abs(Lo.std() - sd)
    p = np.exp(g_lp[idx]); kls = (p * (g_lp[idx] - sub.l2[name][idx])).sum(-1).mean(); klo = (p * (g_lp[idx] - orig.l2[name][idx])).sum(-1).mean()
    return dict(loss=1 - float(np.clip(d_sub / max(d_orig, 1e-12), 0, 1)), pred=1 - float(np.clip(kls / max(klo, 1e-12), 0, 1)),
                mu=float(mu), sd=float(sd), Ls=float(Ls.mean()), SDs=float(Ls.std()), d_orig=float(d_orig), kls=float(kls), klo=float(klo))


def retain(sub, orig, name, idx):
    n = len(orig.z[name]); y = orig.y if n == len(orig.y) else orig.arg[name]
    a_s = (sub.arg[name][idx] == y[idx]).mean(); a_o = (orig.arg[name][idx] == y[idx]).mean()
    po = np.exp(orig.l2[name][idx]); k = (po * (orig.l2[name][idx] - sub.l2[name][idx])).sum(-1).mean()
    return dict(acc_ratio=float(np.clip(a_s / a_o, 0, 1)), kl=float(k), retain=float(np.clip(a_s / a_o, 0, 1) / (1 + k)), acc_sub=float(a_s), acc_orig=float(a_o))


def score(sub, orig, bank, h: Hyp, idx=None, idxc=None, detail=False):
    N = len(orig.y); idx = np.arange(N) if idx is None else idx
    nc = len(orig.z[h.ctrl]); idxc = np.arange(nc) if idxc is None else idxc
    ex = group(sub, orig, *bank.gold(h.lev, h.K, h.gamma, 0, h.rho, h.bg), 'exact', idx)
    al = group(sub, orig, *bank.gold(h.lev + h.dlev, h.K, h.gamma, 4, h.rho, h.bg), h.alt, idx)
    if h.q9 is not None:               # meme recomposition du jeu de controle que dans Fast.ret (remarque du critique, tour 1)
        assert nc == N, 'q9 exige un jeu de controle aligne sur les images a oublier (pas mixes)'
        m = np.isin(orig.y[idxc], C9); a, b = idxc[~m], idxc[m]; idxc = np.concatenate([a, b[:int(round(len(a) * h.q9 / (1 - h.q9)))]])
    r = retain(sub, orig, h.ctrl, idxc)
    loss = (ex['loss'] + al['loss']) / 2; pred = (ex['pred'] + al['pred']) / 2
    out = dict(score=loss * pred * r['retain'], loss=loss, pred=pred, retain=r['retain'])
    if detail: out.update(ex=ex, al=al, r=r)
    return out


class Fast:
    """Meme calcul que score(), avec mise en cache des termes par groupe : une grille d'hypotheses reutilise les memes morceaux."""
    def __init__(self, orig, bank, draws): self.o, self.b, self.d = orig, bank, draws; self.cg = {}; self.cr = {}

    def grp(self, name, sub, key, setname, di):
        k = (name, key, setname, di)
        if k not in self.cg:
            r = group(sub, self.o, *self.b.gold(*key), setname, self.d[di][0]); self.cg[k] = (r['loss'], r['pred'])
        return self.cg[k]

    def ret(self, name, sub, ctrl, di, q9=None):
        k = (name, ctrl, di, q9)
        if k not in self.cr:
            i = self.d[di][1] if len(self.o.z[ctrl]) != len(self.o.y) else self.d[di][0]
            if q9 is not None:        # recompose le jeu de controle : part q9 d'images des classes C9
                assert len(self.o.z[ctrl]) == len(self.o.y), 'q9 exige un jeu de controle aligne sur les images a oublier (pas mixes)'
                m = np.isin(self.o.y[i], C9); a, b = i[~m], i[m]; i = np.concatenate([a, b[:int(round(len(a) * q9 / (1 - q9)))]])
            self.cr[k] = retain(sub, self.o, ctrl, i)['retain']
        return self.cr[k]

    def score(self, name, sub, h):
        """-> (moyenne, ecart-type) du score sur les tirages du sous-ensemble public, et moyennes des trois facteurs."""
        out = []
        for di in range(len(self.d)):
            le, pe = self.grp(name, sub, (h.lev, h.K, h.gamma, 0, h.rho, h.bg), 'exact', di); la, pa = self.grp(name, sub, (min(h.lev + h.dlev, 5), h.K, h.gamma, 4, h.rho, h.bg), h.alt, di)
            r = self.ret(name, sub, h.ctrl, di, h.q9); out.append(((le + la) / 2 * (pe + pa) / 2 * r, (le + la) / 2, (pe + pa) / 2, r))
        a = np.array(out); return float(a[:, 0].mean()), float(a[:, 0].std()), a[:, 1:].mean(0)


def public_draws(N, nc, n=12, frac=0.30, seed=0):
    """Tirages du sous-ensemble public (30 %) : memes indices pour exact et copies (la copie suit sa source)."""
    rng = np.random.default_rng(seed)
    return [(np.sort(rng.choice(N, int(frac * N), replace=False)), np.sort(rng.choice(nc, int(frac * nc), replace=False))) for _ in range(n)]


def score_pub(sub, orig, bank, h, draws):
    """Score attendu sur le public : moyenne et ecart-type sur les tirages du sous-ensemble."""
    s = np.array([score(sub, orig, bank, h, i, ic if len(orig.z[h.ctrl]) != len(orig.y) else i)['score'] for i, ic in draws])
    return float(s.mean()), float(s.std())


def selftest():
    rng = np.random.default_rng(0); N, C = 600, 100; y = rng.integers(0, C, N)

    def mk(conf, flip, seed):
        r = np.random.default_rng(seed); z = r.normal(0, 1, (N, C)); t = y.copy(); w = r.random(N) < flip; t[w] = (t[w] + 1) % C; z[np.arange(N), t] += conf; return z
    names = ['exact'] + SETS_ALT + ['mixes']
    orig = Model({k: mk(14, 0, 1) for k in names}, y)

    class B:
        lev = np.arange(6)
        def gold(self, lev, K, gamma, d0=0, rho=0.0, bg=None):
            z = np.stack([mk(6, 0.15, 10 + k + d0) for k in range(K)]) * gamma; return ce(z, y), np.log(np.exp(lsm(z, T)).mean(0))
    b = B(); h = Hyp(K=1)
    s = score(orig, orig, b, h); assert s['score'] == 0 and abs(s['retain'] - 1) < 1e-12, s
    g = Model({k: mk(6, 0.15, 10) for k in ['exact']} | {k: mk(6, 0.15, 14) for k in SETS_ALT} | {'mixes': orig.z['mixes']}, y)
    s = score(g, orig, b, Hyp(K=1, ctrl='mixes')); assert abs(s['loss'] - 1) < 1e-9 and abs(s['pred'] - 1) < 1e-9 and abs(s['score'] - 1) < 1e-9, s
    print('SELFTEST OK : sub = orig -> 0 ; sub = gold unique et controle intact -> 1')


if __name__ == '__main__':
    selftest()
