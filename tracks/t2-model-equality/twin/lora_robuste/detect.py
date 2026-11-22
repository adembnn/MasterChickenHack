#!/usr/bin/env python
"""Caractéristiques par guichet à partir des relectures par token (rfeat.py), pour N premières réponses.
Toutes les stats ne dépendent que de la base locale exacte tempérée : invariantes à la température par construction (T* ajusté).
"""
import json, os
import numpy as np

BINS = [0.0, 0.3, 1.5, 99.0]   # tranches d'entropie de la base à T=1


def label(m):
    v = m.get("variant", "?")
    if v in ("base", "S"):
        return "copy"
    if v.startswith("lora:") or v in ("T2", "T3", "T4"):
        s = {"T2": "0.3", "T3": "0.7", "T4": "1.5"}.get(v) or v.split("@")[1]
        return "lora" + s
    if v == "?":
        return "?"
    return "quant"


def disg(m):
    d = m.get("disguise") or {}
    k = d.get("kind", "?")
    T = d.get("T")
    return k, T


def load(path):
    z = np.load(path)
    meta = json.load(open(path.replace(".npz", ".meta.json")))
    return z, meta


def fitT(ll, temps):
    """ll : (..., nT) log-vraisemblance sur la grille ; renvoie (max, T*) avec raffinement parabolique."""
    i = np.argmax(ll, -1)
    best = np.take_along_axis(ll, i[..., None], -1)[..., 0]
    return best, temps[i]


WM_GRID = [(0.0, 0.0)] + [(g, d) for g in (0.25, 0.5) for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)]


def wm_loglik(py, p1, top, gamma, delta):
    """log-vraisemblance par token sous « base tempérée + filigrane KGW (gamma, delta) », clé INCONNUE marginalisée :
    couleur du token tiré et du token dominant tirées Bernoulli(gamma), reste de la masse verte à la fraction gamma."""
    if delta == 0:
        return np.log(py + 1e-30)
    e = np.exp(delta) - 1.0
    top = top[:, None]
    # token tiré = token dominant
    Lt = gamma * py * (1 + e) / (1 + e * (py + gamma * (1 - py))) + (1 - gamma) * py / (1 + e * gamma * (1 - py))
    rest = np.clip(1 - py - p1, 0, 1)
    Ln = 0.0
    for gy in (0, 1):
        for g1 in (0, 1):
            pr = (gamma if gy else 1 - gamma) * (gamma if g1 else 1 - gamma)
            Ln = Ln + pr * py * (1 + e * gy) / (1 + e * (gy * py + g1 * p1 + gamma * rest))
    return np.log(np.where(top, Lt, Ln) + 1e-30)


def wm_feats(lp, p1, top, H1, temps):
    py = np.exp(lp.astype(np.float64)); p1 = p1.astype(np.float64)
    best = (-1e300, None, None)
    ll0 = None
    for (g, d) in WM_GRID:
        L = wm_loglik(py, p1, top, g, d)
        tot = L.sum(0)
        if d == 0:
            ll0 = tot.max()
        j = int(np.argmax(tot))
        if tot[j] > best[0]:
            best = (tot[j], (g, d, j), L)
    (g, d, j), L = best[1], best[2]
    lr = 0.0
    for b in range(3):
        mb = (H1 >= BINS[b]) & (H1 < BINS[b + 1])
        if mb.sum() < 5:
            continue
        s = L[mb].sum(0)
        lr += s.max() - s[j]
    return {"wm_gain": float(2 * (best[0] - ll0)), "wm_gamma": g, "wm_delta": d, "T_wm": float(temps[j]), "LRw": float(2 * lr)}


def guichet_feats(z, gi, ctx, nmax, temps, rng=None):
    m = (z["gi"] == gi) & (z["ctx"] == ctx) & (z["slot"] < nmax)
    if m.sum() == 0:
        return None
    lp, H, V, H1, rank, slot = z["lp"][m], z["H"][m], z["V"][m], z["H1"][m], z["rank"][m], z["slot"][m]
    ll = lp.sum(0)
    best, Ts = fitT(ll, temps)
    j = int(np.argmax(ll))
    f = {"T": float(Ts), "ntok": int(m.sum()), "nresp": int(len(np.unique(slot)))}
    # calibration stratifiée : LR (T par tranche vs T commun)
    lr = 0.0
    for b in range(3):
        mb = (H1 >= BINS[b]) & (H1 < BINS[b + 1])
        if mb.sum() < 5:
            f[f"z{b}"] = 0.0; f[f"T{b}"] = float(Ts); continue
        llb = lp[mb].sum(0)
        bb, Tb = fitT(llb, temps)
        lr += bb - llb[j]
        f[f"T{b}"] = float(Tb)
        r = (lp[mb, j] + H[mb, j]).sum(); s = np.sqrt(V[mb, j].sum() + 1e-9)
        f[f"z{b}"] = float(r / s)
    f["LR"] = float(2 * lr)
    f["dT"] = f["T0"] - f["T2"]
    # par réponse : dispersion des résidus (une LoRA de style bouge des réponses entières)
    rr = []
    for s_ in np.unique(slot):
        ms = slot == s_
        rr.append((lp[ms, j] + H[ms, j]).sum() / np.sqrt(V[ms, j].sum() + 1e-9))
    rr = np.array(rr)
    f["resp_var"] = float(rr.var())          # ~1 sous l'hypothèse nulle
    f["resp_min"] = float(rr.min())
    f.update(wm_feats(lp, z["p1"][m], rank == 0, H1, temps))
    f["LRmin"] = min(f["LR"], f["LRw"])
    f["top1"] = float((rank == 0).mean())
    f["rank10"] = float((rank >= 10).mean())
    f["len"] = float(m.sum() / max(1, len(np.unique(slot))))
    return f


def world_table(path, nmax, ctx_mode="auto"):
    z, meta = load(path)
    temps = z["temps"]
    rows = []
    for gi, m in enumerate(meta):
        fn = guichet_feats(z, gi, 0, nmax, temps)
        if fn is None:
            continue
        fe = guichet_feats(z, gi, 1, nmax, temps) if m.get("ext") else None
        fo = guichet_feats(z, gi, 2, nmax, temps) if (m.get("disguise") or {}).get("system") else None
        if ctx_mode == "none" or fe is None:
            f, used = fn, "none"
        elif ctx_mode == "ext":
            f, used = fe, "ext"
        else:  # auto : contexte au LR minimal (une extraction ratée ne peut que perdre)
            f, used = (fe, "ext") if fe["LR"] < fn["LR"] else (fn, "none")
        if ctx_mode == "orc" and fo is not None:
            f, used = fo, "orc"
        k, T = disg(m)
        rows.append(dict(f, gid=m["gid"], lab=label(m), kind=k, Tgen=T, ctx=used, has_ext=fe is not None,
                         world=os.path.basename(path).replace(".npz", "")))
    # version relative : écart à la médiane des guichets du monde (T* < 1), corrige un biais commun serveur/gabarit
    keys = ["LR", "LRw", "LRmin", "wm_gain", "z0", "z1", "z2", "dT", "resp_var", "top1", "rank10"]
    ref = [r for r in rows if r["T"] < 1.0] or rows
    for k in keys:
        med = float(np.median([r[k] for r in ref]))
        mad = float(np.median([abs(r[k] - med) for r in ref])) * 1.4826 + 1e-6
        for r in rows:
            r["rel_" + k] = (r[k] - med) / mad
    return rows
