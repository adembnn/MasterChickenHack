"""Images pleines : combien gagnerait-on si l'on savait retirer les FAUX trous (zones predites propres qui ne touchent aucun vrai trou) ?
Plafond calcule avec la verite, sur les cartes hors-pli. S'il est gros, un test par le decodeur (effacer la zone et regarder si la marge baisse)
pourrait trier vrais et faux trous. Sinon, les erreurs sont des trous manques ou mal dessines, et ce test ne sert a rien."""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from asm import *
tr, te, S, notes = discover(); P = populations(tr, te)
for m, base in (("trustmark", "reel/pred"), ("stegastamp", "reel/pred"), ("pixelseal", "pixelseal/G")):
    if base not in S[m]: continue
    rows = tr[m]; G = np.stack([r["gt"] for r in rows]); L = S[m][base][0]; I = P[m]["tr"]; M = L > 0; sc0, sc1, sc2 = [], [], []; nf = nt = miss = 0; a_f = []
    for i in np.where(I)[0]:
        pred_clean = ~M[i]; true_clean = ~G[i]; lab, n = ndi.label(pred_clean); keep = np.zeros_like(pred_clean); tl, tn = ndi.label(true_clean)
        for k in range(1, n + 1):
            c = lab == k
            if (c & true_clean).any(): keep |= c; nt += 1
            else: nf += 1; a_f.append(c.mean())
        miss += sum(1 for k in range(1, tn + 1) if not ((tl == k) & pred_clean).any())
        sc0.append(miou(M[i][None], G[i][None])[0]); sc1.append(miou((~keep)[None], G[i][None])[0])
        # second plafond : en plus, chaque vrai trou touche est rendu parfaitement (borne haute de tout post-traitement de forme)
        perfect = np.zeros_like(pred_clean)
        for k in range(1, tn + 1):
            t = tl == k
            if (t & pred_clean).any(): perfect |= t
        sc2.append(miou((~perfect)[None], G[i][None])[0])
    print(f"{m:10s} pleines n={int(I.sum())} | carte au seuil 0 : {np.mean(sc0):.3f} | faux trous retires (oracle) : {np.mean(sc1):.3f} | + trous touches rendus parfaits : {np.mean(sc2):.3f}"
          f" | composantes predites propres : {nt} vraies, {nf} fausses (aire mediane des fausses {100 * np.median(a_f) if a_f else 0:.1f} % de l'image) | vrais trous jamais touches : {miss}")
