"""Renomme une partie des copies deguisees (gen_disg2.py) pour les donner a gl_reader.py comme loi nulle des directions :
fk-<T>-<j>, j = 0..N-1 (gl_reader garde j < --kdir). On garde --per faux endpoints par deguisement. Session t2-lecteur.
Usage : mkdgn.py dg_F1.jsonl dgn_F1.jsonl 2"""
import sys, json
src, dst, per = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
keys = sorted({(r["disg"], r["endpoint"]) for r in rows if int(r["endpoint"].split("-")[-1]) % 10 < per})
ren = {k: "fk-%.2f-%03d" % (float(k[1].split("-")[1]), j) for j, k in enumerate(keys)}
with open(dst, "w", encoding="utf-8") as f:
    for r in rows:
        k = (r["disg"], r["endpoint"])
        if k in ren:
            r = dict(r, endpoint=ren[k])
            r.pop("disg")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump({v: k[0] for k, v in ren.items()}, open(dst + ".map.json", "w"), indent=0)
print(len(ren), "faux endpoints ->", dst)
