#!/usr/bin/env python
"""Réponses réelles converties (endpoint/prompt/response/kind) -> format guichets {gid, family, variant, disguise, responses,
prompt_idx}. responses[0] = sonde d'extraction. Usage : python conv.py in.jsonl fam.json out_prefix [truth.csv]"""
import collections, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probes_v4
plan = [s["prompt"] for s in probes_v4.plan()]
pos = collections.defaultdict(list)
for i, p in enumerate(plan):
    pos[p].append(i)
rows = [json.loads(l) for l in open(sys.argv[1])]
fam = json.load(open(sys.argv[2]))["families"]
truth = {}
if len(sys.argv) > 4:
    for l in open(sys.argv[4]).read().splitlines()[1:]:
        e, f, m, k = l.split(",", 3); truth[e] = (m, k)
by = collections.defaultdict(list)
for r in rows:
    by[r["endpoint"]].append(r)
out = collections.defaultdict(list)
for e, rs in sorted(by.items()):
    used = collections.Counter(); resp, pidx = [None], [0]
    for r in rs:
        k = used[r["prompt"]]; used[r["prompt"]] += 1
        if r["prompt"] not in pos or k >= len(pos[r["prompt"]]):
            continue
        i = pos[r["prompt"]][k]
        if i == 0:
            resp[0] = r["response"]
        else:
            resp.append(r["response"]); pidx.append(i)
    m, kind = truth.get(e, ("?", "?"))
    out[fam.get(e, "smol360")].append({"gid": e, "family": fam.get(e, "smol360"), "variant": m, "disguise": {"kind": kind, "system": None},
                                       "responses": resp, "prompt_idx": pidx, "finish": ["stop"] * len(resp)})
for f, G in out.items():
    open(f"{sys.argv[3]}_{f}.jsonl", "w").write("".join(json.dumps(g) + "\n" for g in G))
    print(f, len(G), [len(g["responses"]) for g in G])
