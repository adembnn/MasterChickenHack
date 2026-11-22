#!/usr/bin/env python3
"""Validateur local T3 (Python standard seulement). Usage :
   python3 validate_t3.py submission.pt [--reference model.pt]
Verifie : extension, taille <= 450 Mo, dict plat nom->tenseur, memes noms et formes que ViT-B/16 100 classes
(ou que --reference, mieux), float32 partout, contigu, stockage non gonfle, aucun NaN/Inf.
Ne verifie PAS : l'accuracy >= 70 % ni le score (il faut torch + GPU : voir selfcheck_t3.py dans le rapport)."""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptlite import PtFile, has_nonfinite_f32

LIMIT = 450 * 1024 * 1024


def expected_spec():
    s = collections.OrderedDict()
    s["cls_token"] = (1, 1, 768); s["pos_embed"] = (1, 197, 768)
    s["patch_embed.proj.weight"] = (768, 3, 16, 16); s["patch_embed.proj.bias"] = (768,)
    for i in range(12):
        p = f"blocks.{i}."
        for k, v in [("norm1.weight", (768,)), ("norm1.bias", (768,)), ("attn.qkv.weight", (2304, 768)),
                     ("attn.qkv.bias", (2304,)), ("attn.proj.weight", (768, 768)), ("attn.proj.bias", (768,)),
                     ("ls1.gamma", (768,)), ("norm2.weight", (768,)), ("norm2.bias", (768,)),
                     ("mlp.fc1.weight", (3072, 768)), ("mlp.fc1.bias", (3072,)), ("mlp.fc2.weight", (768, 3072)),
                     ("mlp.fc2.bias", (768,)), ("ls2.gamma", (768,))]:
            s[p + k] = v
    s["norm.weight"] = (768,); s["norm.bias"] = (768,); s["head.weight"] = (100, 768); s["head.bias"] = (100,)
    return s


def main(argv):
    if not argv: sys.exit(__doc__)
    path = argv[0]; ref = argv[argv.index("--reference") + 1] if "--reference" in argv else None
    errs, warns = [], []
    if os.path.splitext(path)[1].lower() not in (".pt", ".pth"): errs.append("extension != .pt/.pth")
    size = os.path.getsize(path)
    if size > LIMIT: errs.append(f"taille {size/2**20:.1f} Mio > 450 Mio")
    try:
        f = PtFile(path)
    except Exception as e:
        sys.exit(f"ECHEC lecture (zip/pickle invalide ou objet interdit) : {e}")
    obj = f.obj
    if not isinstance(obj, dict): sys.exit(f"ECHEC : racine = {type(obj).__name__}, il faut un dict nom->tenseur")
    if "state_dict" in obj and len(obj) < 5: errs.append("cle 'state_dict' : c'est un checkpoint, pas un state_dict nu")
    if ref:
        r = PtFile(ref).obj; spec = collections.OrderedDict((k, v.size) for k, v in r.items())
        print(f"reference : {ref} ({len(spec)} cles)")
    else:
        spec = expected_spec(); warns.append("spec integree (hypothese timm) ; preferer --reference model.pt")
    missing, extra = [k for k in spec if k not in obj], [k for k in obj if k not in spec]
    if missing: errs.append(f"cles manquantes ({len(missing)}): {missing[:5]}")
    if extra: errs.append(f"cles en trop ({len(extra)}): {extra[:5]} (prefixe module./_orig_mod. ? fusion LoRA oubliee ?)")
    total = bad_nf = 0
    for k, t in obj.items():
        if not hasattr(t, "storage"): errs.append(f"{k}: pas un tenseur"); continue
        if k in spec and tuple(spec[k]) != t.size: errs.append(f"{k}: forme {t.size} != {tuple(spec[k])}")
        if t.dtype != "float32": errs.append(f"{k}: dtype {t.dtype} (float32 exige)"); continue
        if not t.contiguous(): warns.append(f"{k}: non contigu")
        if t.storage.numel != t.numel: warns.append(f"{k}: stockage {t.storage.numel} > {t.numel} elements (vue ? taille gonflee)")
        total += t.numel
        if has_nonfinite_f32(f.storage_bytes(t)): errs.append(f"{k}: NaN ou Inf"); bad_nf += 1
    print(f"{len(obj)} tenseurs, {total:,} parametres, {size/2**20:.1f} Mio")
    for w in warns: print("AVERTISSEMENT", w)
    for e in errs: print("ERREUR", e)
    print("RESULTAT:", "INVALIDE" if errs else "VALIDE (format) ; accuracy non verifiee ici")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
