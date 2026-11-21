#!/usr/bin/env python
"""Journaux bruts collecte_v4 (type raw, raw_body JSON) -> JSONL {"endpoint","prompt","response","kind"}.
Usage : python conv_collecte.py sortie.jsonl journal1.jsonl [journal2.jsonl ...]
"""
import collections, json, sys

out, n = open(sys.argv[1], "w"), collections.Counter()
for path in sys.argv[2:]:
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("type") != "raw" or r.get("http_status") != 200 or r.get("dry_run"):
            continue
        try:
            c = json.loads(r["raw_body"])["choices"][0]["message"]["content"]
        except Exception:
            continue
        if c:
            out.write(json.dumps({"endpoint": r["endpoint"], "prompt": r["prompt"], "response": c, "kind": r.get("kind")}) + "\n")
            n[r["endpoint"]] += 1
print(dict(sorted(n.items())))
