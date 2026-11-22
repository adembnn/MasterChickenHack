"""Point 3 : quelles questions portent le plus d'information par token sur une LoRA faible ? (aucune requete)
On apprend la direction sur la LoRA forte des ref (ref-07, toutes ses reponses), puis, question par question, on mesure le gain de
log-vraisemblance par token (dose 1,0, a la T* de chaque guichet) pour la LoRA 0,3 (ref-09), la LoRA 0,7 (ref-04) et les 5 copies.
Pouvoir de separation d'une question = (gain LoRA 0,3 - gain moyen des copies) / ecart type des copies, par reponse.
Usage : prompt_info.py + arguments de gl_reader.py"""
import sys, json, statistics, collections
exec(open("gl_reader.py", encoding="utf-8").read().split('R = {"TS": TS')[0])
base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
B = profile(base, {e: ENC[e] for e in eps})
Tst = {e: TS[max(range(len(TS)), key=lambda i: B[e][i])] for e in eps}
def item_ll(model, e):
    out = []
    with torch.no_grad():
        for i in range(0, len(ENC[e]), 16):
            ch = ENC[e][i:i + 16]; ids, att = batch([c + y for c, y in ch])
            lg = model(input_ids=ids, attention_mask=att).logits
            for j, (c, y) in enumerate(ch):
                z = lg[j, len(c) - 1:len(c) + len(y) - 1].float() / Tst[e]; tg = ids[j, len(c):len(c) + len(y)]
                out.append((float(torch.log_softmax(z, -1).gather(1, tg[:, None]).sum()), len(y)))
    return out
L0 = {e: item_ll(base, e) for e in eps}
SRC = "ref-07"
pm, tau = train(base, ENC[SRC], Tst[SRC], 0); set_scale(pm, 1.0)
L1 = {e: item_ll(pm, e) for e in eps if e != SRC}
COP = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]
byq = collections.defaultdict(lambda: collections.defaultdict(list))
for e in L1:
    for (l1, n), (l0, _), p in zip(L1[e], L0[e], PR[e]):
        byq[p][e].append(((l1 - l0) / n, n))
rows = []
for p, d in byq.items():
    cop = [g for e in COP for g, n in d.get(e, [])]
    if len(cop) < 4 or "ref-09" not in d: continue
    mu, sd = statistics.mean(cop), statistics.pstdev(cop) or 1e-9
    g3 = statistics.mean(g for g, n in d["ref-09"]); g7 = statistics.mean(g for g, n in d.get("ref-04", [(0, 1)]))
    ntok = statistics.mean(n for e in d for g, n in d[e])
    rows.append(((g3 - mu) / sd, (g7 - mu) / sd, g3 - mu, ntok, p))
rows.sort(reverse=True)
print(f"direction apprise sur {SRC} ; {len(rows)} questions ; separation = (gain LoRA - gain copies) / ecart type des copies, par reponse")
print("sep 0,3 | sep 0,7 | ecart nat/token | tokens | question")
for s3, s7, d, n, p in rows: print(f"{s3:+5.1f} | {s7:+5.1f} | {d:+.4f} | {n:5.0f} | {p[:110]}")
print(f"moyenne de la separation LoRA 0,3 : {statistics.mean(r[0] for r in rows):+.2f} ; top tiers {statistics.mean(r[0] for r in rows[:len(rows) // 3]):+.2f} ; bas tiers {statistics.mean(r[0] for r in rows[-(len(rows) // 3):]):+.2f}")
print("FINI")
