#!/usr/bin/env python
"""T2 - trois leviers de partition, appliques sur les SORTIES TEXTE (seules) des guichets :

  1. Decodage sous contrainte d'inventaire : un bonus/malus de taille de grappe (grande classe
     4-6, fp8 0-2, une LoRA par force -> singleton, reste singleton) pese sur le clustering de
     correlation, pour voir si un prior volontairement FAUX (+-1 sur la taille de la grande classe)
     coute plus que 0,05 de MCC.
  2. Vote entre detecteurs (profil de NLL sous un lecteur local unique + recouvrement lexical),
     clustering de correlation, et diagnostic des triangles incoherents (A=B, B=C, A!=C) : le
     coupable probable est l'endpoint qui apparait dans le plus de triangles incoherents.
  3. Decodage MBR : 200 rééchantillonnages bootstrap des prompts partages, partition a chaque
     tirage, puis partition "medoide" (celle qui ressemble le plus a la moyenne de toutes).

Aucune requete vers l'API du hackathon : opere sur un JSONL de reponses deja recueillies (le banc
CPU jumeau world_cpu.jsonl, ou les vraies reponses ref-*/ep-* deja collectees).
"""
import argparse, csv, itertools, json, math, random, sys, time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
READER_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


# --------------------------------------------------------------------- donnees
def load_jsonl_world(path):
    """endpoint -> {prompt_idx: (prompt, response)}"""
    by_ep = defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        by_ep[d["endpoint"]][d["prompt_idx"]] = (d["prompt"], d["response"])
    return by_ep


def load_truth_csv(path):
    truth = {}
    for row in csv.DictReader(open(path, newline="", encoding="utf-8")):
        truth[row["endpoint"]] = (row["family"], row["model"])
    return truth


def common_prompt_idx(by_ep):
    idx = None
    for ep, d in by_ep.items():
        s = set(d)
        idx = s if idx is None else (idx & s)
    return sorted(idx)


# --------------------------------------------------------------------- lecteur local (jamais les poids caches)
def reader_nll_matrix(by_ep, idx, model_id=READER_MODEL, max_ctx=None, bs=12, log=print):
    """Pour chaque guichet, NLL moyenne PAR TOKEN de chaque reponse relue sous le SEUL lecteur local
    (texte retokenise, jamais les ids internes du guichet). endpoint -> [nll_i pour i in idx]."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    log(f"lecteur local charge ({model_id}), jamais partage avec un guichet")

    out = {}
    names = sorted(by_ep)
    with torch.no_grad():
        for ep in names:
            rows, plens = [], []
            for i in idx:
                prompt, resp = by_ep[ep][i]
                ctx = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                               add_generation_prompt=True)
                p_ids = tok(ctx, add_special_tokens=False)["input_ids"]
                g_ids = tok(resp, add_special_tokens=False)["input_ids"]
                if max_ctx:
                    g_ids = g_ids[:max_ctx]
                rows.append(p_ids + g_ids)
                plens.append(len(p_ids))
            nlls = [float("nan")] * len(rows)
            for s in range(0, len(rows), bs):
                chunk, clen = rows[s:s + bs], plens[s:s + bs]
                L = max(len(r) for r in chunk)
                ids = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(chunk), L), dtype=torch.long)
                for j, r in enumerate(chunk):
                    ids[j, :len(r)] = torch.tensor(r)
                    att[j, :len(r)] = 1
                logp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), dim=-1)
                for j, r in enumerate(chunk):
                    pl = clen[j]
                    n = len(r) - pl
                    if n <= 0:
                        continue
                    sl = logp[j, pl - 1:len(r) - 1]
                    tgt = ids[j, pl:len(r)]
                    lp_t = sl.gather(-1, tgt[:, None]).squeeze(-1)
                    nlls[s + j] = float((-lp_t).mean())
            out[ep] = nlls
    del model
    return out


# --------------------------------------------------------------------- detecteurs (independants)
def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return 0.0
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / math.sqrt(va * vb)


def ngrams(text, n=3):
    toks = text.lower().split()
    return set(tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))) or {tuple(toks)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    inter, union = len(a & b), len(a | b)
    return inter / union if union else 0.0


def detector_nll_profile(nll_mat, names):
    """Detecteur A : correlation du PROFIL de NLL sous le lecteur, prompt par prompt. Deux guichets des
    memes poids devraient rater/reussir sur les memes prompts, quel que soit le deguisement de decodage."""
    return {(a, b): pearson(nll_mat[a], nll_mat[b]) for a, b in itertools.combinations(names, 2)}


def detector_lexical(by_ep, idx, names):
    """Detecteur B : recouvrement lexical (3-grammes de mots) moyenne sur les prompts partages. Signal
    independant du A (texte brut, pas de relecture), complementaire quand une famille entiere est loin
    du lecteur local."""
    grams = {ep: [ngrams(by_ep[ep][i][1]) for i in idx] for ep in names}
    out = {}
    for a, b in itertools.combinations(names, 2):
        vals = [jaccard(ga, gb) for ga, gb in zip(grams[a], grams[b])]
        out[(a, b)] = sum(vals) / len(vals)
    return out


def zscore(d):
    vals = list(d.values())
    m, n = sum(vals) / len(vals), len(vals)
    v = sum((x - m) ** 2 for x in vals) / n
    sd = math.sqrt(v) or 1.0
    return {k: (x - m) / sd for k, x in d.items()}


def ensemble_scores(detA, detB, names):
    zA, zB = zscore(detA), zscore(detB)
    return {p: (zA[p] + zB[p]) / 2 for p in itertools.combinations(names, 2)}


# --------------------------------------------------------------------- lever 2a : triangles incoherents
def triangle_diagnosis(ens, names, tau=0.0):
    same = lambda a, b: ens[(a, b)] > tau if (a, b) in ens else ens[(b, a)] > tau
    culprit_count = defaultdict(int)
    bad_triangles = []
    for a, b, c in itertools.combinations(names, 3):
        sab, sbc, sac = same(a, b), same(b, c), same(a, c)
        votes_same = sum([sab, sbc, sac])
        if votes_same in (1, 2):  # transitivement incoherent (0 ou 3 "same" votes = coherent)
            # coupable probable : l'endpoint dont les deux aretes disagree avec l'edge restante
            scores = {a: abs(ens.get((a, b), ens.get((b, a), 0))) + abs(ens.get((a, c), ens.get((c, a), 0))),
                      b: abs(ens.get((a, b), ens.get((b, a), 0))) + abs(ens.get((b, c), ens.get((c, b), 0))),
                      c: abs(ens.get((a, c), ens.get((c, a), 0))) + abs(ens.get((b, c), ens.get((c, b), 0)))}
            # le coupable est celui dont les DEUX aretes vers les 2 autres sont les PLUS FAIBLES en magnitude
            culprit = min(scores, key=scores.get)
            culprit_count[culprit] += 1
            bad_triangles.append((a, b, c, culprit))
    return bad_triangles, culprit_count


# --------------------------------------------------------------------- clustering de correlation (greedy)
def correlation_clustering(ens, names, size_prior=None, lam=0.0):
    """Fusion agglomerative gloutonne : a chaque pas, fusionne la paire de grappes dont le score moyen
    (+ bonus de prior de taille) est le plus eleve, tant qu'il est positif."""

    def log_prior(s):
        if size_prior is None:
            return 0.0
        return size_prior(s)

    clusters = {n: {n} for n in names}
    cid = {n: n for n in names}

    def cluster_score(c1, c2):
        pairs = [ens[(a, b)] if (a, b) in ens else ens[(b, a)] for a in clusters[c1] for b in clusters[c2]]
        raw = sum(pairs) / len(pairs)
        bonus = lam * (log_prior(len(clusters[c1]) + len(clusters[c2])) - log_prior(len(clusters[c1])) - log_prior(len(clusters[c2])))
        return raw + bonus

    while True:
        best, best_score = None, -1e18
        ids = list(clusters)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sc = cluster_score(ids[i], ids[j])
                if sc > best_score:
                    best_score, best = sc, (ids[i], ids[j])
        if best is None or best_score <= 0:
            break
        c1, c2 = best
        clusters[c1] |= clusters[c2]
        del clusters[c2]
    labels = {}
    for lab, members in enumerate(clusters.values()):
        for m in members:
            labels[m] = lab
    return labels


def default_size_prior(target_big=5.0, single_bonus=0.6, var=3.0):
    def f(s):
        if s == 1:
            return single_bonus
        return -((s - target_big) ** 2) / (2 * var)
    return f


# --------------------------------------------------------------------- lever 3 : MBR bootstrap-medoide
def bootstrap_mbr(by_ep, idx, nll_mat, names, n_boot=200, seed=0):
    rnd = random.Random(seed)
    nK = len(idx)
    grams_full = {ep: [ngrams(by_ep[ep][i][1]) for i in idx] for ep in names}
    partitions, adj_list = [], []
    for b in range(n_boot):
        sample = [rnd.randrange(nK) for _ in range(nK)]
        nllA = {ep: [nll_mat[ep][i] for i in sample] for ep in names}
        detA = {(a, c): pearson(nllA[a], nllA[c]) for a, c in itertools.combinations(names, 2)}
        detB = {(a, c): sum(jaccard(grams_full[a][i], grams_full[c][i]) for i in sample) / nK
                for a, c in itertools.combinations(names, 2)}
        ens = ensemble_scores_from(detA, detB, names)
        labels = correlation_clustering(ens, names)
        partitions.append(labels)
        adj = {p: (1 if labels[p[0]] == labels[p[1]] else 0) for p in itertools.combinations(names, 2)}
        adj_list.append(adj)
    pairs = list(itertools.combinations(names, 2))
    stability = {p: sum(a[p] for a in adj_list) / n_boot for p in pairs}
    mean_adj = stability
    best_i, best_d = 0, 1e18
    for i, adj in enumerate(adj_list):
        d = sum((adj[p] - mean_adj[p]) ** 2 for p in pairs)
        if d < best_d:
            best_d, best_i = d, i
    return partitions[best_i], stability


def ensemble_scores_from(detA, detB, names):
    return ensemble_scores(detA, detB, names)


# --------------------------------------------------------------------- score MCC (pairs, comme score_practice.py)
def mcc_over_pairs(labels, truth, names):
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(names, 2):
        t = truth[a] == truth[b]
        g = labels[a] == labels[b]
        tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = max(0.0, (tp * tn - fp * fn) / d) if d else 0.0
    return mcc, dict(tp=tp, fp=fp, fn=fn, tn=tn)


def no_wrongly_isolated(labels, truth, names):
    """Un vrai singleton (seul de son modele) etiquete seul : ok. Un membre d'une VRAIE classe >1 isole a tort."""
    size_true = defaultdict(int)
    for n in names:
        size_true[truth[n]] += 1
    size_lab = defaultdict(int)
    for n in names:
        size_lab[labels[n]] += 1
    bad = []
    for n in names:
        if size_true[truth[n]] > 1 and size_lab[labels[n]] == 1:
            bad.append(n)
    return bad


# --------------------------------------------------------------------- CLI
def cmd_bench(a):
    t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
    by_ep = load_jsonl_world(a.jsonl)
    truth = load_truth_csv(a.truth)
    names = sorted(by_ep)
    idx = common_prompt_idx(by_ep)
    log(f"{len(names)} guichets, {len(idx)} prompts communs")

    nll_mat = reader_nll_matrix(by_ep, idx, log=log)
    detA = detector_nll_profile(nll_mat, names)
    detB = detector_lexical(by_ep, idx, names)
    ens = ensemble_scores(detA, detB, names)
    log("detecteurs A (profil NLL) et B (recouvrement lexical) calcules")

    # ---- levier 2 : triangles incoherents + clustering de correlation (sans prior) ----
    bad_tri, culprits = triangle_diagnosis(ens, names)
    log(f"triangles incoherents : {len(bad_tri)} / {math.comb(len(names), 3)}")
    top_culprits = sorted(culprits.items(), key=lambda kv: -kv[1])[:5]
    log("coupables probables (top 5) :", top_culprits)
    labels_v2 = correlation_clustering(ens, names)
    mcc_v2, conf_v2 = mcc_over_pairs(labels_v2, truth, names)
    bad_v2 = no_wrongly_isolated(labels_v2, truth, names)
    log(f"levier 2 (vote+clustering, sans prior) : MCC={mcc_v2:.3f} {conf_v2} isoles_a_tort={bad_v2}")

    # ---- levier 1 : prior d'inventaire correct vs volontairement faux (+-1 sur la grande classe) ----
    true_big = max(sum(1 for n in names if truth[n] == m) for m in set(truth.values()))
    for lam in (0.0, 0.5):
        for shift, tag in ((0, "correct"), (-1, "faux-1"), (1, "faux+1")):
            prior = default_size_prior(target_big=true_big + shift)
            labels = correlation_clustering(ens, names, size_prior=prior, lam=lam)
            mcc, conf = mcc_over_pairs(labels, truth, names)
            log(f"levier 1 lambda={lam} prior_taille={true_big+shift} ({tag}) : MCC={mcc:.3f} {conf}")

    # ---- levier 3 : MBR bootstrap-medoide ----
    labels_mbr, stab = bootstrap_mbr(by_ep, idx, nll_mat, names, n_boot=a.n_boot)
    mcc_mbr, conf_mbr = mcc_over_pairs(labels_mbr, truth, names)
    stable_frac = sum(1 for v in stab.values() if v >= 0.8 or v <= 0.2) / len(stab)
    log(f"levier 3 (MBR, {a.n_boot} tirages) : MCC={mcc_mbr:.3f} {conf_mbr} "
        f"part_paires_stables(>=80%)={stable_frac:.2f}")

    out = {"names": names, "truth": {n: list(truth[n]) for n in names}, "ens": {f"{k[0]}|{k[1]}": v for k, v in ens.items()},
           "labels_v2": labels_v2, "labels_mbr": labels_mbr, "stability": {f"{k[0]}|{k[1]}": v for k, v in stab.items()},
           "bad_triangles": len(bad_tri), "culprits": dict(culprits)}
    json.dump(out, open(a.out, "w"), indent=1)
    log("FIN ->", a.out)


def cmd_apply_real(a):
    """Applique les 3 leviers aux vraies reponses deja collectees (ep-* ou ref-*), sans verite connue :
    rapporte juste la partition finale, la stabilite et les triangles incoherents (les candidats a
    revoir), pour comparer a une partition de reference (ex. celle de la methode cross-NLL/histogramme
    deja soumise)."""
    t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
    raw = defaultdict(lambda: defaultdict(dict))
    for line in open(a.jsonl, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("type") != "answer" or d.get("kind") != "reading":
            continue
        raw[d["endpoint"]][d["probe_id"]] = (d["prompt"], d["response"])
    by_ep = {ep: {i: v for i, v in enumerate(sorted(pbs.items()))} for ep, pbs in raw.items()}
    by_ep = {ep: {i: (pid_pr[1][0], pid_pr[1][1]) for i, pid_pr in enumerate(sorted(pbs.items()))}
             for ep, pbs in raw.items()}
    names = sorted(by_ep)
    idx = common_prompt_idx(by_ep)
    log(f"{len(names)} guichets reels, {len(idx)} prompts communs : {names}")
    if len(idx) < 3:
        sys.exit("pas assez de prompts communs pour ce fichier")

    nll_mat = reader_nll_matrix(by_ep, idx, log=log)
    detA = detector_nll_profile(nll_mat, names)
    detB = detector_lexical(by_ep, idx, names)
    ens = ensemble_scores(detA, detB, names)

    bad_tri, culprits = triangle_diagnosis(ens, names)
    top_culprits = sorted(culprits.items(), key=lambda kv: -kv[1])[:8]
    log(f"triangles incoherents : {len(bad_tri)} / {math.comb(len(names), 3)}; coupables :", top_culprits)

    labels_v2 = correlation_clustering(ens, names)
    labels_mbr, stab = bootstrap_mbr(by_ep, idx, nll_mat, names, n_boot=a.n_boot)
    out = {"names": names, "labels_clustering": labels_v2, "labels_mbr": labels_mbr,
           "stability": {f"{k[0]}|{k[1]}": v for k, v in stab.items()}, "culprits": dict(culprits)}
    json.dump(out, open(a.out, "w"), indent=1)
    log("FIN ->", a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench")
    b.add_argument("--jsonl", default=str(HERE / "world_cpu.jsonl"))
    b.add_argument("--truth", default=str(HERE / "world_cpu_truth.csv"))
    b.add_argument("--out", default=str(HERE / "results_partition_levers_bench.json"))
    b.add_argument("--n_boot", type=int, default=200)
    b.set_defaults(func=cmd_bench)

    r = sub.add_parser("apply_real")
    r.add_argument("--jsonl", required=True)
    r.add_argument("--out", default=str(HERE / "results_partition_levers_real.json"))
    r.add_argument("--n_boot", type=int, default=200)
    r.set_defaults(func=cmd_apply_real)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
