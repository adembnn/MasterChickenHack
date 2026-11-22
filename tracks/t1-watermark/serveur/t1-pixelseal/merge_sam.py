"""Fusionne les morceaux sam_<split>_<methode>.s<i>of<n>.npz (ou .part.npz) en sam_<split>_<methode>.npz. Usage : python merge_sam.py <dossier> <split> <methode> [partiel]"""
import sys, glob, numpy as np
d, split, method = sys.argv[1:4]; partial = len(sys.argv) > 4
fs = sorted(glob.glob(f"{d}/sam_{split}_{method}.s*of*.npz")); done = [f for f in fs if not f.endswith(".part.npz")]
use = fs if partial else done; out = {}
for f in sorted(use, key=lambda f: f.endswith(".part.npz"), reverse=True):      # les fichiers complets écrasent les partiels
    z = np.load(f); out.update({k: z[k] for k in z.files})
np.savez_compressed(f"{d}/sam_{split}_{method}.npz", **out); print(split, method, "morceaux complets", len(done), "| images", len(out) // 2)
