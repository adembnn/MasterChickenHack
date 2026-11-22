#!/usr/bin/env python
"""Régime où tous les endpoints échantillonnent : statistiques de distribution sur les réponses fermées (stdlib).

Un item = (sonde fermée, rang de la mini-question). Chaque endpoint donne R réponses par item (R répétitions).
  self(A)    : probabilité que deux réponses de A au même item soient égales (taux de collision interne). Plus la
               température est haute, plus il est bas : sert à repérer l'endpoint « température 1,0 ».
  cross(A,B) : probabilité qu'une réponse de A et une réponse de B au même item soient égales.
  D(A,B)     : self(A) + self(B) - 2 cross(A,B), calculé par item. C'est un estimateur sans biais de la distance
               quadratique entre les deux distributions de réponses : son espérance est 0 quand A et B ont la même
               distribution, quel que soit le hasard de l'échantillonnage.
  z(A,B)     : D moyen divisé par son erreur type (écart type entre items / racine du nombre d'items). Test
               auto-calibré : z au-delà de 2 ou 3 = distributions différentes. Aucun seuil appris sur le banc.
  mode(A,B)  : part des items dont la réponse la plus fréquente est la même (insensible à la température).
"""
import math, statistics
from collections import Counter
from features import closed_answers, CLOSED_KINDS


def answers(ep):
    """{(pid, item): [réponse par répétition]} pour un Endpoint (features.Endpoint)."""
    out = {}
    for pid, texts in ep.texts.items():
        k = ep.kinds.get(pid, "open")
        if k not in CLOSED_KINDS or len(texts) < 2:
            continue
        per = [closed_answers(k, t, pid) for t in texts]
        keys = set().union(*[set(p) for p in per])
        for it in keys:
            out[(pid, it)] = [p.get(it, "?") for p in per]
    return out


def _self(v):
    n = len(v)
    c = Counter(v)
    return sum(x * (x - 1) for x in c.values()) / (n * (n - 1)) if n > 1 else float("nan")


def self_rate(a):
    v = [_self(x) for x in a.values() if len(x) > 1]
    return statistics.mean(v) if v else float("nan")


def compare(a, b):
    ds, modes = [], []
    for key, va in a.items():
        vb = b.get(key)
        if not vb or len(va) < 2 or len(vb) < 2:
            continue
        ca, cb = Counter(va), Counter(vb)
        cross = sum(ca[x] * cb.get(x, 0) for x in ca) / (len(va) * len(vb))
        ds.append(_self(va) + _self(vb) - 2 * cross)
        modes.append(ca.most_common(1)[0][0] == cb.most_common(1)[0][0])
    n = len(ds)
    if n < 5:
        return {"D": float("nan"), "z": float("nan"), "mode": float("nan"), "n": n}
    d = statistics.mean(ds)
    se = statistics.pstdev(ds) / math.sqrt(n) or 1e-9
    return {"D": d, "z": d / se, "mode": statistics.mean(modes), "n": n}


def compare_text(ta, tb, kinds, which=("open", "meta")):
    """Test à deux échantillons sur les réponses libres (texte entier), par sonde répétée au moins 2 fois de chaque côté.
    Noyau = cosinus des sacs de mots. D = k(A,A') + k(B,B') - 2 k(A,B) est un estimateur sans biais de la MMD au carré
    (maximum mean discrepancy : distance entre deux distributions de textes), d'espérance 0 si les distributions sont
    égales. z = D moyen / erreur type entre sondes. dlen : même construction sur la log-longueur des réponses."""
    from features import _cos
    ds, dl = [], []
    for pid, a in ta.items():
        b = tb.get(pid)
        if kinds.get(pid) not in which or not b or len(a) < 2 or len(b) < 2:
            continue
        kaa = statistics.mean(_cos(a[i], a[j]) for i in range(len(a)) for j in range(i + 1, len(a)))
        kbb = statistics.mean(_cos(b[i], b[j]) for i in range(len(b)) for j in range(i + 1, len(b)))
        kab = statistics.mean(_cos(x, y) for x in a for y in b)
        ds.append(kaa + kbb - 2 * kab)
        la = statistics.mean(math.log(len(x) + 5) for x in a)
        lb = statistics.mean(math.log(len(x) + 5) for x in b)
        dl.append(la - lb)
    n = len(ds)
    if n < 4:
        return {"zt": float("nan"), "zlen": float("nan"), "nt": n}
    zt = statistics.mean(ds) / (statistics.pstdev(ds) / math.sqrt(n) or 1e-9)
    zl = statistics.mean(dl) / (statistics.pstdev(dl) / math.sqrt(n) or 1e-9)
    return {"zt": zt, "zlen": abs(zl), "nt": n}
