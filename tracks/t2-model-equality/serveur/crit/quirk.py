import json, sys, collections, re
def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
ep = [r for r in load("collecte_v4_ep_bloc0_20261121T193826Z.jsonl") if r.get("type") == "answer"]
tests = {"F3": ("date differente de 2026 dans P000", lambda r: r["probe_id"] == "P000" and bool(re.search(r"Today Date", r["response"])) and "2026" not in r["response"][:400] and bool(re.search(r"Nov \d{4}", r["response"]))),
         "F3b": ("ecrit 1866", lambda r: r["probe_id"] == "P000" and "1866" in r["response"]),
         "F3c": ("cite Today Date", lambda r: r["probe_id"] == "P000" and "Today Date" in r["response"]),
         "F2": ("refus (sorry) dans P000", lambda r: r["probe_id"] == "P000" and "sorry" in r["response"].lower()),
         "F2b": ("Claude dans P001", lambda r: r["probe_id"] == "P001" and "Claude" in r["response"]),
         "F1": ("Claude dans P001", lambda r: r["probe_id"] == "P001" and "Claude" in r["response"]),
         "F1b": ("refus (sorry) dans P000", lambda r: r["probe_id"] == "P000" and "sorry" in r["response"].lower())}
for key, (name, fn) in tests.items():
    fam = key[:2]
    nl = load(f"null_{fam}.jsonl")
    by = collections.defaultdict(lambda: [0, 0])
    for r in nl:
        if r["probe_id"] == ("P001" if "P001" in name else "P000"):
            by[r["T"]][1] += 1
            by[r["T"]][0] += fn(r)
    print(key, name, "| jumeau :", " ".join(f"T={T}: {a}/{b}" for T, (a, b) in sorted(by.items())))
nl = load("null_F3.jsonl")
ys = collections.Counter(m for r in nl if r["probe_id"] == "P000" for m in re.findall(r"Nov (\d{4})", r["response"]))
print("annees ecrites par le jumeau Llama dans P000 :", ys.most_common(8))
