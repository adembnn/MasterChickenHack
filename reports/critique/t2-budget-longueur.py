"""t2-budget : mots par réponse selon le type de question (ep, blocs 0 + 1). Plus la réponse est longue, plus il y a de tokens lus par requête. 0 requête."""
import json, re, collections, statistics as st, importlib.util
spec = importlib.util.spec_from_file_location("bi", r"E:\EAC\reports\critique\t2-budget-info.py"); bi = importlib.util.module_from_spec(spec); spec.loader.exec_module(bi)
ep = bi.load(["collecte_v4_ep_bloc0_20261121T193826Z.jsonl", "collecte_v4_ep_bloc1_20261122T011020Z.jsonl", "collecte_v4_ep_bloc1_20261122T021208Z.jsonl"])
t = collections.defaultdict(list); sd = {}
for (e, p), xs in ep.items():
    for x in xs: t[bi.qtype(p)].append(len(bi.words(x)))
for k, v in sorted(t.items(), key=lambda kv: -st.mean(kv[1])):
    print(f"{k:9s} n={len(v):4d} mots moyen {st.mean(v):6.1f} médiane {st.median(v):6.1f}  >=170 mots : {sum(x>=170 for x in v)/len(v):.2f}")
