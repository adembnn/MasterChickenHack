#!/usr/bin/env python
"""Rapport de validation du test réel n°1 (bloc 0) : répond point par point, chiffres à l'appui, à ce que le test
valide ou invalide dans la méthode v4. Lecture de fichiers seulement (journal ep, sorties de run_ep.sh et run_real.sh).

Seuils fixés AVANT de voir les données ep :
  - « ne colle à aucun candidat » : meilleure NLL/token > 2,0 (max observé sur les vrais ref sous le bon jumeau : 1,84,
    ref-08 à T=1,0) OU écart au 2e candidat < 0,03 ;
  - « sort à T=1,0 comme ref-08 » : T* de lecture ≥ 1,0 ;
  - « system prompt / persona extrait » : la réponse à la sonde commence par « You are » ou contient « assistant named »,
    « your name is », « persona », « system prompt » ; « écho de la consigne » si elle reprend « Repeat all the text ».

Usage : python validate_test1.py --ep_journal <glob> --ep_dir ep_bloc0 --ref_dir real_bloc0 --ref_assign ep_refsim/assign.json \
          --ref_twin SmolLM2-360M-Instruct [--public_score 0.xx] --out validation_test1.md
"""
import argparse
import glob
import json
import statistics as st
from collections import Counter
from pathlib import Path

KEYS_SYS = ("assistant named", "your name is", "persona", "system prompt")


def load_answers(pattern):
    out = {}
    for path in sorted(glob.glob(pattern)):
        if path.endswith("_dryrun.jsonl"):
            continue
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("type") == "answer" and r.get("response") is not None:
                out.setdefault(r["endpoint"], {})[(r["probe_id"], r["repeat"])] = r
    return out


def q(xs, f):
    xs = sorted(xs)
    return xs[int(f * (len(xs) - 1))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep_journal", required=True)
    ap.add_argument("--ep_dir", required=True)
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--ref_assign", required=True)
    ap.add_argument("--ref_twin", default="SmolLM2-360M-Instruct")
    ap.add_argument("--ref_dump", default=None, help="dump decide des ref (défaut : <ref_dir>/dump_<ref_twin>.json)")
    ap.add_argument("--public_score", type=float, default=None)
    ap.add_argument("--out", default="validation_test1.md")
    a = ap.parse_args()
    L = []
    W = L.append
    ans = load_answers(a.ep_journal)
    eps = sorted(ans)
    asg = json.load(open(Path(a.ep_dir) / "assign.json"))
    ref_asg = json.load(open(a.ref_assign))
    twin_full = {m.split("/")[-1]: m for m in asg["nll"]}
    dumps = {p.stem.replace("dump_", ""): json.load(open(p)) for p in Path(a.ep_dir).glob("dump_*.json")}
    ep_dump = {e: d for fam in dumps.values() for e, d in fam.items()}
    ref_dump = json.load(open(a.ref_dump or (Path(a.ref_dir) / f"dump_{a.ref_twin}.json")))
    W(f"# T2 · Validation du test réel n°1 (bloc 0, {len(eps)} ep, {sum(len(v) for v in ans.values())} réponses)\n")

    # (1) échantillonnage
    W("## 1. Les ep échantillonnent-ils ? (question P002 envoyée 2 fois)")
    rep = {}
    for e in eps:
        r = [ans[e][k]["response"] for k in [("P002", 0), ("P002", 1)] if k in ans[e]]
        rep[e] = None if len(r) < 2 else (r[0] != r[1])
    n_s = sum(1 for v in rep.values() if v)
    n_g = sum(1 for v in rep.values() if v is False)
    W(f"- échantillonnent : **{n_s}** ; déterministes : **{n_g}** ; incomplets : {sum(1 for v in rep.values() if v is None)}")
    W("- déterministes : " + (", ".join(e for e, v in rep.items() if v is False) or "aucun"))
    W(f"- comparaison ref : 9/9 échantillonnent.\n")

    # (2) familles
    W("## 2. Familles (jumeau de NLL minimale) et garde « aucun candidat ne colle »")
    full_ref = next(m for m in ref_asg["nll"] if m.endswith(a.ref_twin))
    ref_nll = list(ref_asg["nll"][full_ref].values())
    fam = asg["family_twin"]
    for m, c in Counter(fam.values()).most_common():
        mem = sorted(e for e in eps if fam.get(e) == m)
        W(f"- **{m.split('/')[-1]}** : {len(mem)} ep : {', '.join(mem)}")
    nofit = [e for e in eps if e in fam and (asg["nll"][fam[e]][e] > 2.0 or asg["margin"][e] < 0.03)]
    W(f"- ne collent à aucun candidat (meilleure NLL > 2,0 ou écart < 0,03) : **{len(nofit)}** "
      + (": " + ", ".join(f"{e} (NLL {asg['nll'][fam[e]][e]:.2f}, écart {asg['margin'][e]:.2f})" for e in nofit) if nofit else ""))
    W(f"- repère ref sous {a.ref_twin} : NLL {min(ref_nll):.2f} à {max(ref_nll):.2f}\n")
    W("| ep | famille | NLL meilleure | écart au 2e |")
    W("|---|---|---|---|")
    for e in eps:
        if e in fam:
            W(f"| {e} | {fam[e].split('/')[-1]} | {asg['nll'][fam[e]][e]:.3f} | {asg['margin'][e]:.3f} |")
    W("")

    # (3) structure
    W("## 3. Structure des groupes par famille (corrigé ref : 1 base + 4 copies = 5 « S », 4 variantes isolées)")
    for name, d in sorted(dumps.items()):
        sizes = Counter(v["label"] for v in d.values())
        W(f"- {name} ({len(d)} ep) : groupe S = {sizes.get('S', 0)} ; variantes : {len([k for k in sizes if k != 'S'])} "
          f"groupe(s), tailles {sorted((v for k, v in sizes.items() if k != 'S'), reverse=True)}")
    W("")

    # (4) extraction
    W("## 4. Sonde d'extraction : system prompt ou persona rendu ?")
    hits, echo = [], []
    for e in eps:
        t = ans[e].get(("P000", 0), {}).get("response", "")
        low = t.lower()
        if t.strip().startswith("You are") or any(k in low for k in KEYS_SYS):
            hits.append(e)
        elif "repeat all the text" in low:
            echo.append(e)
    W(f"- prompt caché probable : **{len(hits)}** : {', '.join(hits) or 'aucun'} ; écho de la consigne : {len(echo)}")
    for e in hits:
        W(f"  - {e} : « {ans[e][('P000', 0)]['response'][:160].replace(chr(10), ' ')} »")
    W("")

    # (5) température
    W("## 5. Température de lecture T* (endpoints qui échantillonnent)")
    Ts = {e: d["T_star"] for e, d in ep_dump.items() if d.get("sampling") and "T_star" in d}
    if Ts:
        W(f"- répartition : {dict(sorted(Counter(Ts.values()).items()))}")
        W(f"- T* ≥ 1,0 (comme ref-08) : {', '.join(f'{e} ({t})' for e, t in sorted(Ts.items()) if t >= 1.0) or 'aucun'}")
    W(f"- repère ref : {dict(sorted(Counter(d['T_star'] for d in ref_dump.values() if 'T_star' in d).items()))} (ref-08 = 1,0)\n")

    # (6) LLR
    W("## 6. Rapport de vraisemblance (t de Welch par variante locale) : ep vs ref")
    W("Variantes locales : fp8 = fausse quantification e4m3 **par ligne** (pas par tenseur), int8, LoRA locales (autre corpus).")
    cols = next((list(d["llr_t"]) for d in ref_dump.values() if "llr_t" in d), [])
    if cols:
        W("| endpoint | " + " | ".join(cols) + " | décision |")
        W("|---|" + "---|" * (len(cols) + 1))
        for e, d in sorted(ref_dump.items()):
            if "llr_t" in d:
                W(f"| {e} | " + " | ".join(f"{d['llr_t'][c]:+.1f}" for c in cols) + f" | {d['label']} |")
        for e, d in sorted(ep_dump.items()):
            if "llr_t" in d:
                W(f"| {e} | " + " | ".join(f"{d['llr_t'].get(c, float('nan')):+.1f}" for c in cols) + f" | {d['label']} |")
        for c in cols:
            rv = [d["llr_t"][c] for d in ref_dump.values() if "llr_t" in d]
            ev = [d["llr_t"][c] for d in ep_dump.values() if "llr_t" in d and c in d["llr_t"]]
            if rv and ev:
                W(f"- {c} : ref médiane {st.median(rv):+.1f} (|t|≥3 : {sum(abs(x) >= 3 for x in rv)}/{len(rv)}) ; "
                  f"ep médiane {st.median(ev):+.1f} (|t|≥3 : {sum(abs(x) >= 3 for x in ev)}/{len(ev)})")
    W("")

    rcols = next((list(d["readers_z"]) for d in ref_dump.values() if "readers_z" in d), [])
    if rcols:
        W("\n**Lecteurs de la règle principale** (z apparié vs endpoints de T* < 1,0 de la même famille) :")
        W("| endpoint | " + " | ".join(rcols) + " | décision |")
        W("|---|" + "---|" * (len(rcols) + 1))
        for src in (ref_dump, ep_dump):
            for e, d in sorted(src.items()):
                if "readers_z" in d:
                    W(f"| {e} | " + " | ".join(f"{d['readers_z'].get(c, float('nan')):+.1f}" for c in rcols) + f" | {d['label']} |")
    W("")

    # (7) ressemblance ep / ref
    W(f"## 7. Les ep de la famille {a.ref_twin} ressemblent-ils aux ref ?")
    fam_eps = [e for e in eps if fam.get(e, "").endswith(a.ref_twin)]
    if fam_eps:
        full = next(m for m in asg["nll"] if m.endswith(a.ref_twin))
        en = [asg["nll"][full][e] for e in fam_eps]
        W(f"- NLL/token sous {a.ref_twin} : ep médiane {st.median(en):.2f} [{min(en):.2f} ; {max(en):.2f}] vs ref médiane "
          f"{st.median(ref_nll):.2f} [{min(ref_nll):.2f} ; {max(ref_nll):.2f}]")
        raw_ep = Path(a.ep_dir) / f"raw_{a.ref_twin}.json"
        raw_ref = Path(a.ref_dir) / f"raw_{a.ref_twin}.json"
        if raw_ep.exists() and raw_ref.exists():
            le = [x for v in json.load(open(raw_ep))["endpoints"].values() for x in v["lengths"]]
            lr = [x for v in json.load(open(raw_ref))["endpoints"].values() for x in v["lengths"]]
            W(f"- longueur des réponses (tokens) : ep médiane {st.median(le):.0f} (p10 {q(le, .1):.0f}, p90 {q(le, .9):.0f}) vs ref "
              f"médiane {st.median(lr):.0f} (p10 {q(lr, .1):.0f}, p90 {q(lr, .9):.0f})")
    else:
        W(f"- aucun ep assigné à {a.ref_twin} : les ref ne sont PAS un banc de calibration direct pour les ep.")
    W("")

    # (8) leaderboard
    W("## 8. Score public après soumission")
    if a.public_score is None:
        W("- à remplir après la soumission (session principale).")
    else:
        W(f"- score public : **{a.public_score:.3f}** ; attendu par la méthode : terme famille correct si familles parmi les candidats, "
          "terme modèle 0 à 0,3 (vrais ref : 0,273).")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
