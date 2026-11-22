def rrc(x, g, lo, hi): return affine(x, g, scale=(lo, hi), flip=True)[0]


def mix_half(pool, B, g):
    """Melanges ou aucune image source ne pese plus de la moitie. pool : indices (cuda) dans fgd."""
    n = len(pool); n1 = B // 3; n2 = B // 3; n3 = B - n1 - n2; H = W = 224

    def pos(m): return torch.randint(0, n, (m,), device=dev, generator=g)

    def other(p): return (p + torch.randint(1, n, (len(p),), device=dev, generator=g)) % n

    def img(p, lo=0.5, hi=1.0): return rrc(to01(fgd[pool[p]]), g, lo, hi)
    p1 = pos(n1); p2 = other(p1); p3 = other(p1)
    a, b, c = img(p1), img(p2), img(p3); lam = U(n1, 0.15, 0.5, g).view(-1, 1, 1, 1); three = (U(n1, 0, 1, g) < 0.5).float().view(-1, 1, 1, 1)
    m1 = three * (0.5 * a + lam * b + (0.5 - lam) * c) + (1 - three) * (0.5 * a + 0.5 * b)
    p1 = pos(n2); p2 = other(p1); a, b = img(p1), img(p2)
    r = U(n2, 0.5, 1.0, g); bw = (r * W).long(); bh = (0.5 / r * H).long()
    x0 = (U(n2, 0, 1, g) * (W - bw + 1)).long(); y0 = (U(n2, 0, 1, g) * (H - bh + 1)).long()
    ii = torch.arange(H, device=dev).view(1, H, 1); jj = torch.arange(W, device=dev).view(1, 1, W)
    mask = ((ii >= y0.view(-1, 1, 1)) & (ii < (y0 + bh).view(-1, 1, 1)) & (jj >= x0.view(-1, 1, 1)) & (jj < (x0 + bw).view(-1, 1, 1))).unsqueeze(1)
    m2 = torch.where(mask, b, a)
    t = [F.interpolate(img(pos(n3), 0.2, 0.5), size=112, mode='bilinear', antialias=True, align_corners=False) for _ in range(4)]
    m3 = torch.cat([torch.cat([t[0], t[1]], 3), torch.cat([t[2], t[3]], 3)], 2)
    return torch.cat([m1, m2, m3]).clamp(0, 1)


