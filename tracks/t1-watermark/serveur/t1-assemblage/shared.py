"""Deux images du jeu partagent-elles des pixels EXACTS (meme scene d'origine, objet greffe pris dans une autre image du jeu, fond propre commun) ?
Si oui, la difference pixel a pixel donnerait le masque exact. Blocs 16x16 non plats, hachage ; collisions entre images differentes,
a la meme position puis a n'importe quelle position."""
import sys, json, hashlib, collections, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
B = 16; items = [("train", r) for r in load_train()] + [("test", r) for r in load_test()]
pos = collections.defaultdict(list); free = collections.defaultdict(list)
for k, (sp, r) in enumerate(items):
    a = load_img(sp, r["image_name"]); blk = a.reshape(512 // B, B, 512 // B, B, 3).transpose(0, 2, 1, 3, 4).reshape(-1, B * B * 3)
    keep = np.where(blk.astype(np.float32).std(1) > 6)[0]
    for j in keep:
        h = hashlib.md5(blk[j].tobytes()).digest()[:8]; pos[(j, h)].append(k); free[h].append((k, j))
    if k % 300 == 0: print(k, len(pos), flush=True)
pairs = collections.Counter()
for (j, h), L in pos.items():
    L = sorted(set(L))
    for x in range(len(L)):
        for y in range(x + 1, len(L)): pairs[(L[x], L[y])] += 1
print("blocs non plats :", len(pos), "| paires d'images partageant au moins un bloc exact a la MEME position :", len(pairs))
for (x, y), n in pairs.most_common(15): print("  ", n, "blocs |", items[x][0], items[x][1]["method"], items[x][1]["image_name"], "|", items[y][0], items[y][1]["method"], items[y][1]["image_name"])
pf = collections.Counter()
for h, L in free.items():
    ims = sorted({k for k, _ in L})
    if 1 < len(ims) <= 6:
        for x in range(len(ims)):
            for y in range(x + 1, len(ims)): pf[(ims[x], ims[y])] += 1
print("paires partageant au moins un bloc exact a une position QUELCONQUE (grille de 16) :", len(pf))
for (x, y), n in pf.most_common(15): print("  ", n, "blocs |", items[x][0], items[x][1]["method"], items[x][1]["image_name"], "|", items[y][0], items[y][1]["method"], items[y][1]["image_name"])
