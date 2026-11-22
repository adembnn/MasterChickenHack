#!/usr/bin/env python
"""Affiche le type détecté et les indices par endpoint. Usage : python dump_kinds.py probes.jsonl monde/*.jsonl"""
import sys, json
from parse import load
from features import build
probes = [json.loads(l) for f in sys.argv[1].split(",") for l in open(f, encoding="utf-8") if l.strip()]
kinds, data = load(sys.argv[2:], sys.argv[1].split(",")[0])
kinds.update({p["id"]: p["kind"] for p in probes})
eps = build(data, kinds, prompts={p["id"]: p["prompt"] for p in probes})
for n in sorted(eps):
    e = eps[n]
    print("%-22s nondet=%.2f fuites=%d recur=%d id=%-28s lp=%s  %r" % (n, e.nondet, e.n_leak, e.recur, ",".join(e.id_words)[:28],
          " ".join("%s:%.1f" % (k, v) for k, v in sorted(e.lp.items())), e.leak[:40]))
