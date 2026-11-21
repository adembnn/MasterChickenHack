"""Cibles S1 : (mu, sigma) selon l'accuracy non-membre supposee A, ajustement (a,b), cibles q (T=1) et q2 (T=2)."""
import math, torch, torch.nn.functional as F

# Hypothese de modele de loss d'un non-membre (NON MESUREE) : bien classes -> loss moyenne 0.12, ecart-type 0.25 ;
# mal classes (surconfiance d'un reseau affine) -> loss moyenne 3.0, ecart-type 1.3. Melange pondere par A.
MC, SC, MW, SW = 0.12, 0.25, 3.0, 1.3


def gold_moments(A):
    mu = A * MC + (1 - A) * MW
    var = A * (SC ** 2 + MC ** 2) + (1 - A) * (SW ** 2 + MW ** 2) - mu ** 2
    return mu, math.sqrt(var)


def fit_ab(z, mu, sd):
    """(a,b) tels que L = softplus(b*z - a) = -log sigmoid(a - b z) ait moyenne mu et ecart-type sd sur les images (z = difficulte normalisee)."""
    zz = z.double().cuda(); best = (1e9, 0, 0)
    for a in torch.linspace(-1, 8, 181).double():
        for b in torch.linspace(0.05, 8, 160).double():
            L = F.softplus(b * zz - a); e = (L.mean() - mu) ** 2 + (L.std() - sd) ** 2
            if e < best[0]: best = (e.item(), a.item(), b.item())
    return best[1], best[2], best[0]


def make_q(p_y, yhat, w):
    """q = p_y e_yhat + (1-p_y) w ; q2 = 0.85 norm(q^(1/2)) + 0.15 w  (T=2 : q^(1/2) est ce que donne softmax(z/2) si softmax(z)=q ; le 0.15 w modelise une moyenne de plusieurs golds)."""
    N = len(yhat); q = (1 - p_y)[:, None] * w; q[torch.arange(N), yhat] += p_y
    h = q.clamp_min(1e-12).sqrt(); h = h / h.sum(1, keepdim=True)
    return q, 0.85 * h + 0.15 * w
