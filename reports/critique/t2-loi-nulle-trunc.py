"""Decodage tronque (top-p / top-k) ou tirage pur ? (aucune requete)
Pour chaque token rendu : c = masse de probabilite du jumeau (a temperature T) placee sur des tokens PLUS probables que lui, et son rang.
Tirage pur a la bonne T : c est a peu pres uniforme, donc ~10 % des tokens ont c > 0,9 et ~5 % c > 0,95. Top-p 0,9 : aucun token au-dela de 0,9.
Top-k 50 : aucun rang > 50. On lit a T* (meilleure temperature) et a T = 1,0. Usage : arguments de gl_reader.py."""
import sys, json
exec(open("gl_reader.py", encoding="utf-8").read().split('R = {"TS": TS')[0])
base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
B = profile(base, {e: ENC[e] for e in eps})
Tst = {e: TS[max(range(len(TS)), key=lambda i: B[e][i])] for e in eps}
print("guichet  T*   tokens | a T* : part c>0,90  c>0,95  c>0,99  rang max  rang>50 | a T=1,0 : c>0,90  c>0,95  c>0,99")
for e in eps:
    cs = {Tst[e]: [], 1.0: []}; rk = []
    with torch.no_grad():
        for i in range(0, len(ENC[e]), 16):
            ch = ENC[e][i:i + 16]; ids, att = batch([c + y for c, y in ch])
            lg = base(input_ids=ids, attention_mask=att).logits
            for j, (c, y) in enumerate(ch):
                z = lg[j, len(c) - 1:len(c) + len(y) - 1].float(); tg = ids[j, len(c):len(c) + len(y)]
                for T in cs:
                    p = torch.softmax(z / T, -1); py = p.gather(1, tg[:, None])
                    cs[T].append(((p * (p > py)).sum(1)).cpu())
                rk.append(((z > z.gather(1, tg[:, None])).sum(1) + 1).cpu())
    r = torch.cat(rk); out = []
    for T in cs:
        c = torch.cat(cs[T]); out.append((float((c > .9).float().mean()), float((c > .95).float().mean()), float((c > .99).float().mean())))
    print(f"{e}  {Tst[e]:.2f}  {len(r):5d} |        {out[0][0]:.3f}   {out[0][1]:.3f}   {out[0][2]:.4f}   {int(r.max()):6d}   {float((r > 50).float().mean()):.4f} |          {out[1][0]:.3f}   {out[1][1]:.3f}   {out[1][2]:.4f}", flush=True)
print("FINI")
