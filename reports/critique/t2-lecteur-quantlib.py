"""Quantifications simulees (poids arrondis puis remis en bf16) des couches lineaires des blocs, tete de sortie et
plongements laisses en pleine precision. Session t2-lecteur, aucune requete.
Schemas : int8_ch (absmax par ligne), int8_t (par tenseur), int4_g128 (symetrique, groupes de 128), nf4_b64 (code NF4 de
bitsandbytes, absmax par bloc de 64), fp8_ch / fp8_t (e4m3, echelle par ligne / par tenseur), fp4_b64 (code FP4 bnb)."""
import torch

NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                    0.7229568362236023, 1.0])
FP4 = torch.tensor([0.0, 0.0052083333, 0.6666667, 1.0, 0.33333334, 0.5, 0.16666667, 0.25,
                    -0.0, -0.0052083333, -0.6666667, -1.0, -0.33333334, -0.5, -0.16666667, -0.25])


def _code(w, code, bs):
    sh = w.shape
    x = w.reshape(-1, bs)
    am = x.abs().amax(1, keepdim=True).clamp_min(1e-12)
    xn = x / am
    c = code.to(w.device)
    idx = (xn[..., None] - c).abs().argmin(-1)
    return (c[idx] * am).reshape(sh)


def qw(w, scheme):
    w = w.float()
    if scheme == "int8_ch":
        s = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
        return (w / s).round().clamp(-127, 127) * s
    if scheme == "int8_t":
        s = w.abs().max().clamp_min(1e-12) / 127
        return (w / s).round().clamp(-127, 127) * s
    if scheme == "int4_g128":
        o, i = w.shape
        x = w.reshape(o, i // 128, 128)
        s = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7
        return ((x / s).round().clamp(-8, 7) * s).reshape(o, i)
    if scheme == "nf4_b64":
        return _code(w, NF4, 64)
    if scheme == "fp4_b64":
        return _code(w, FP4, 64)
    if scheme in ("fp8_ch", "fp8_t"):
        s = (w.abs().amax(1, keepdim=True) if scheme == "fp8_ch" else w.abs().max()).clamp_min(1e-12) / 448
        return (w / s).to(torch.float8_e4m3fn).float() * s
    raise ValueError(scheme)


@torch.no_grad()
def fake_quant(model, scheme):
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".layers." in name:
            # par morceaux de lignes pour limiter la memoire (nf4 : argmin sur 16 codes)
            W = mod.weight.data
            step = W.shape[0] if scheme.endswith("_t") else max(1, (1 << 22) // W.shape[1])
            for r in range(0, W.shape[0], step):
                W[r:r + step] = qw(W[r:r + step], scheme).to(W.dtype)
            n += 1
    return n
