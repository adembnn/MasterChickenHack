"""Réseau du décodeur local dense PixelSeal (partagé par loc.py et predict.py)."""
import torch, torch.nn as nn, torch.nn.functional as F

class Loc(nn.Module):
    def __init__(s, det, d=160, res=256, agree=1, ppinit=1, hr=0):
        super().__init__(); s.det = det; s.res = res; s.agree = agree; s.ppinit = ppinit; s.hr = hr; ch = [96, 192, 384, 768]
        s.lat = nn.ModuleList([nn.Conv2d(c, d, 1) for c in ch])
        s.sm = nn.ModuleList([nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GroupNorm(8, d), nn.GELU()) for _ in ch])
        s.bit = nn.Conv2d(d, 256, 1); s.ppscale = nn.Parameter(torch.ones(1))
        if ppinit: nn.init.zeros_(s.bit.weight); nn.init.zeros_(s.bit.bias)
        s.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, 48, 3, 2, 1), nn.GELU())
        if hr:                                        # branche 'stéganalyse' : pleine résolution, aucune réduction avant 6 couches
            s.hr_in = nn.Conv2d(6, hr, 3, padding=1)
            s.hr_blocks = nn.ModuleList([nn.Sequential(nn.GroupNorm(8, hr), nn.GELU(), nn.Conv2d(hr, hr, 3, padding=dl, dilation=dl)) for dl in (1, 1, 2, 2, 4, 1)])
            s.hr_out = nn.Sequential(nn.GroupNorm(8, hr), nn.GELU(), nn.Conv2d(hr, 48, 1))
        s.head = nn.Sequential(nn.Conv2d(d + 48 + 2, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(), nn.Conv2d(128, 128, 3, padding=1), nn.GELU(), nn.Conv2d(128, 1, 1))
    def forward(s, x, sgn):                           # x B x 3 x 512 x 512 dans [0,1] ; sgn B x 256 (+1/-1)
        xi = F.interpolate(x, size=(s.res, s.res), mode="bilinear", align_corners=False, antialias=True) if s.res != x.shape[-1] else x
        h = xi * 2 - 1; fs = []
        for dl, st in zip(s.det.convnext.downsample_layers, s.det.convnext.stages):
            h = st(dl(h)); fs.append(h)
        p = None
        for i in range(3, -1, -1):
            l = s.lat[i](fs[i]); p = l if p is None else l + F.interpolate(p, size=l.shape[-2:], mode="bilinear", align_corners=False); p = s.sm[i](p)
        L = s.bit(p).float()
        if s.ppinit:                                  # départ = votes par case de la tête d'origine (8x8), agrandis ; le réseau apprend à les rendre locaux
            lin = s.det.pixel_decoder.linear; emb = s.det.pixel_decoder.output_upscaling(fs[3])
            pp = torch.einsum("bchw,oc->bohw", emb.float(), lin.weight[1:].float()) + lin.bias[1:].float()[None, :, None, None]
            L = L + s.ppscale * F.interpolate(pp, size=L.shape[-2:], mode="bilinear", align_corners=False)
        t = torch.tanh(L / 2)
        a1 = (t * sgn[:, :, None, None]).mean(1, keepdim=True); a2 = t.abs().mean(1, keepdim=True)
        if not s.agree: a1 = a1 * 0; a2 = a2 * 0
        up = lambda z: F.interpolate(z, size=(128, 128), mode="bilinear", align_corners=False)
        e = s.stem(x * 2 - 1)
        if s.hr:
            with torch.autocast('cuda', enabled=False):   # première couche en fp32 : un niveau (1/255) doit rester lisible
                xf = x.float(); hp = (xf - F.avg_pool2d(F.pad(xf, (1, 1, 1, 1), mode='replicate'), 3, 1)) * 64
                h2 = s.hr_in(torch.cat([xf * 2 - 1, hp], 1))
            for b in s.hr_blocks: h2 = h2 + b(h2)
            e = e + F.avg_pool2d(s.hr_out(h2), 4)
        m = s.head(torch.cat([up(p), e, up(a1).to(p.dtype), up(a2).to(p.dtype)], 1))[:, 0].float()
        return m, L, a1[:, 0]
