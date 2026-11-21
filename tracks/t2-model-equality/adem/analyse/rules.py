#!/usr/bin/env python
"""Décodeur à règles (v4), sans apprentissage : peu de paramètres, donc peu de risque de coller à une famille.

Entrées : endpoints d'UNE famille, statistiques par paire F[frozenset((a, b))], type par endpoint (P, S, T).
  1. Les endpoints P (déterministes, sans prompt système détecté) sont regroupés par identité de texte (statistique de
     la v3 de Titouan) : lien moyen coupé à tau_pp. C'est le seul cas où la comparaison exacte est valide.
  2. Groupe central = celui de l'endpoint P qui est le plus proche voisin du plus grand nombre d'endpoints (vote),
     à défaut le plus proche en moyenne. Dans le corrigé, les
     variantes de poids (quantification, LoRA) sont toutes dérivées de la base : la base est au centre.
  3. Filigrane : un endpoint P isolé dont le biais par paire d'étiquettes s'écarte du groupe central d'au moins tau_w
     (slp : écart type, entre paires d'étiquettes, des écarts signés) est traité comme une copie filigranée du groupe
     central.
  4. Chaque endpoint S est rattaché au groupe P dont il est le plus proche (sim), le groupe central l'emportant si
     l'écart est inférieur à `margin`. Chaque endpoint T est rattaché au groupe central (aucune statistique mesurée ne
     voit à travers la température 1,0 avec 100 réponses).
"""
import statistics


def nanmean(v):
    v = [x for x in v if x == x]
    return statistics.mean(v) if v else 0.0


SIMS = {
    "ident": lambda f: nanmean([f["ident"]]),
    "ident+pref": lambda f: nanmean([f["ident"], f["pref"]]),
    "fermées": lambda f: nanmean([f["m_ab5"], f["m_nb5"], f["m_q1"], f["m_word"], f["m_lp"]]),
    "mixte": lambda f: nanmean([f["ident"], f["pref"], f["m_ab5"], f["m_nb5"], f["m_q1"]]),
    "bow": lambda f: nanmean([f["bow"]]),
}


def linkage(eps, S, cut):
    cl = [[e] for e in eps]
    while len(cl) > 1:
        best, bi = -1, None
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                s = statistics.mean(S[frozenset((a, b))] for a in cl[i] for b in cl[j])
                if s > best:
                    best, bi = s, (i, j)
        if best < cut:
            break
        i, j = bi
        cl[i] += cl[j]
        del cl[j]
    return {e: k for k, c in enumerate(cl) for e in c}


def decode(eps, F, kinds, tau_pp=0.8, margin=0.05, tau_w=0.25, sim="ident+pref", attach_s=True, attach_t=True,
           detect_w=True, hub_votes="nn4", wstat="slp"):
    eps = list(eps)
    simf = SIMS[sim]
    S = {pr: simf(f) for pr, f in F.items()}
    ident = {pr: (f["ident"] if f["ident"] == f["ident"] else 0.0) for pr, f in F.items()}
    ps = [e for e in eps if kinds[e] == "P"]
    lab, nxt = {}, 0
    info = {"hub": None, "w": [], "attach": {}}
    if ps:
        lab = linkage(ps, ident, tau_pp)
        nxt = max(lab.values()) + 1
        if hub_votes in ("nn", "nn4"):  # chaque endpoint vote pour l'endpoint P le plus proche de lui
            votes = {e: 0 for e in ps}
            # nn4 : un vote par statistique de proximité (4 statistiques), pour ne dépendre d'aucune en particulier
            for sf in ([SIMS[n] for n in ("ident", "ident+pref", "fermées", "mixte")] if hub_votes == "nn4" else [simf]):
                for v in eps:
                    cand = [o for o in ps if o != v]
                    if cand:
                        votes[max(cand, key=lambda o: sf(F[frozenset((v, o))]))] += 1
            hub_ep = max(ps, key=lambda e: (votes[e], statistics.mean([S[frozenset((e, o))] for o in eps if o != e] or [0])))
        else:
            voters = [e for e in eps if kinds[e] != "T"] if hub_votes == "PS" else ps
            hub_ep = max(ps, key=lambda e: statistics.mean([S[frozenset((e, o))] for o in voters if o != e] or [0]))
        hub = lab[hub_ep]
        info["hub"] = hub_ep
        groups = {}
        for e, k in lab.items():
            groups.setdefault(k, []).append(e)
        if detect_w:
            for e in ps:
                if lab[e] != hub and len(groups[lab[e]]) == 1:
                    d = nanmean([F[frozenset((e, o))][wstat] for o in groups[hub]])
                    if d >= tau_w:
                        info["w"].append(e)
            for e in info["w"]:
                groups[lab[e]].remove(e)
                lab[e] = hub
            groups = {k: g for k, g in groups.items() if g}
            groups[hub] = [e for e in ps if lab[e] == hub]
    for e in eps:
        k = kinds[e]
        if k == "P":
            continue
        if not ps or (k == "S" and not attach_s) or (k == "T" and not attach_t):
            lab[e] = nxt
            nxt += 1
        elif k == "T" and attach_t != "nearest":
            lab[e] = hub
        else:
            anchors = {g: [o for o in members if o not in info["w"]] or members for g, members in groups.items()}
            score = {g: statistics.mean(S[frozenset((e, o))] for o in members) for g, members in anchors.items()}
            best = max(score, key=score.get)
            lab[e] = hub if score[best] - score[hub] <= margin else best
            info["attach"][e] = (lab[e] == hub, round(score[best] - score[hub], 3))
    return lab, info


FAMILY_SIMS = ("ident", "ident+pref", "fermées", "mixte")


def family_partition(names, F, cut=0.4):
    """Familles : lien moyen sur une proximité normalisée par endpoint (s(e,o) divisé par la meilleure proximité de e,
    moyenne des deux sens, moyenne de 4 statistiques). Mesuré sur 2 familles locales seulement : seuil fragile."""
    names = list(names)
    if len(names) < 3:
        return {e: 0 for e in names}
    sims = [SIMS[n] for n in FAMILY_SIMS]
    best = {(i, e): max(sf(F[frozenset((e, o))]) for o in names if o != e) or 1e-9
            for i, sf in enumerate(sims) for e in names}
    S = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            f = F[frozenset((x, y))]
            S[frozenset((x, y))] = sum(0.5 * (sf(f) / best[(k, x)] + sf(f) / best[(k, y)])
                                       for k, sf in enumerate(sims)) / len(sims)
    return linkage(names, S, cut)


def decode_sampled(eps, F, E, zcut=3.0, attach=True):
    """Régime où tous les endpoints échantillonnent (aucun endpoint P). Plus de comparaison exacte possible : on teste
    l'égalité des distributions de réponses (analyse/sampled.py). Pour chaque paire, z = le plus grand de trois tests à
    deux échantillons (réponses fermées par item, textes libres, longueur). Le centre est l'endpoint sans prompt système
    dont le z médian contre les autres est le plus bas ; un endpoint sans prompt système rejoint le centre si son z
    contre lui est sous zcut, sinon il forme un modèle à part ; les endpoints à prompt système vont au centre.
    Mesuré (rapport, section 3.7) : à 100 réponses, seuls les écarts de poids grossiers dépassent z = 3."""
    import sampled
    eps = list(eps)
    A = {e: sampled.answers(E[e]) for e in eps}
    nz = lambda v: -9.0 if v != v else v
    Z = {}
    for i, x in enumerate(eps):
        for y in eps[i + 1:]:
            c = sampled.compare(A[x], A[y])
            t = sampled.compare_text(E[x].texts, E[y].texts, E[x].kinds)
            Z[frozenset((x, y))] = max(nz(c["z"]), nz(t["zt"]), nz(t["zlen"]))
    ps = [e for e in eps if not E[e].sys]
    if not ps:
        return {e: i for i, e in enumerate(eps)}, {"hub": None, "w": [], "attach": {}}
    hub = min(ps, key=lambda e: statistics.median([Z[frozenset((e, o))] for o in ps if o != e] or [0]))
    lab, nxt, att = {}, 1, {}
    for e in eps:
        if e == hub or (E[e].sys and attach) or (not E[e].sys and Z[frozenset((e, hub))] < zcut):
            lab[e] = 0
        else:
            lab[e] = nxt
            nxt += 1
        if e != hub:
            att[e] = (lab[e] == 0, round(Z[frozenset((e, hub))], 1))
    return lab, {"hub": hub, "w": [], "attach": att}
