"""t3-verificateur : controle de fichier + sorties sur les images EXACTES seulement, par petits lots (quand la carte est pleine).
Ecrit out/NOM.npz (cle 'exact' seule : suffit a pgc.py, pas a vstat.py) et out/NOM.check.json. usage : python vquick.py NOM=chemin.pt ..."""
import sys, os, json, numpy as np, torch
import vlogits as V
if __name__ == '__main__':
    fg = np.load(V.ROOT + '/forget_images.npz')['images']; ref = torch.load(V.ROOT + '/model.pt', map_location='cpu', weights_only=True)
    for a in sys.argv[1:]:
        name, path = a.split('=', 1)
        if os.path.exists(f'{V.OUT}/{name}.npz'): print(name, 'deja fait', flush=True); continue
        m, r = V.check_file(path, ref); z = V.predict(m, fg, bs=25); r['fp16_vs_fp32_max_abs_logit'] = -1.0; r['fp16_vs_fp32_argmax_diff'] = -1; r['logits_finite'] = bool(np.isfinite(z).all()); r['exact_only'] = True
        np.savez(f'{V.OUT}/{name}.npz', exact=z); json.dump(r, open(f'{V.OUT}/{name}.check.json', 'w'), indent=1)
        y = np.load(V.OUT + '/orig.npz')['exact'].argmax(1); zz = z.astype(np.float64); zz -= zz.max(1, keepdims=True); ce = -(zz - np.log(np.exp(zz).sum(1, keepdims=True)))[np.arange(len(y)), y]
        print(f"{name} sha {r['sha256'][:16]} strict={'All keys' in r['strict_load']} f32={r['all_float32']} fini={r['finite']} modifies={r['n_changed']} gamma={len(r['changed_gamma'])} | m {ce.mean():.2f} s {ce.std():.2f} bascules {(z.argmax(1) != y).mean():.3f}", flush=True)
        del m; torch.cuda.empty_cache()
