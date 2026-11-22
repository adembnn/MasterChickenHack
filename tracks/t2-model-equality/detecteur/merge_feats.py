"""Fusionne les caractéristiques de deux blocs (mêmes guichets, mêmes colonnes) : bloc0 (slots 0-24) + bloc1 (slots 25-49).
python merge_feats.py A.npz B.npz OUT.npz   (copie aussi A.meta.json -> OUT.meta.json ; l'ordre des guichets doit être identique)"""
import json, sys
import numpy as np
a, b, out = sys.argv[1:4]
da, db = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
ma, mb = json.load(open(a.replace(".npz", ".meta.json"))), json.load(open(b.replace(".npz", ".meta.json")))
assert [m["gid"] for m in ma] == [m["gid"] for m in mb], "ordre des guichets différent"
assert list(da["cols"]) == list(db["cols"])
np.savez_compressed(out, X=np.concatenate([da["X"], db["X"]]), gi=np.concatenate([da["gi"], db["gi"]]),
                    ri=np.concatenate([da["ri"], db["ri"]]), cols=da["cols"])
json.dump(ma, open(out.replace(".npz", ".meta.json"), "w"))
print("fusion", out, da["X"].shape[0] + db["X"].shape[0], "lignes")
