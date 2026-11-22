import json
with open("null_ref4.jsonl", "w", encoding="utf-8") as f:
    for l in open("null_ref_k104.jsonl", encoding="utf-8"):
        r = json.loads(l); a, T, k = r["endpoint"].split("-"); r["endpoint"] = "fk-%s-%03d" % (T, int(k) // 4); f.write(json.dumps(r, ensure_ascii=False) + chr(10))
with open("ref_pool100.jsonl", "w", encoding="utf-8") as f:
    for p in ("collecte_v4_ref_bloc0_20261121T191611Z.jsonl", "ref_frais_1.jsonl", "ref_frais_2.jsonl", "ref_frais_3.jsonl"):
        for l in open(p, encoding="utf-8"):
            if l.strip() and json.loads(l).get("type", "answer") == "answer":
                f.write(l.rstrip(chr(10)) + chr(10))
print("ok")
