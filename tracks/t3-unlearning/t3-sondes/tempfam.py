"""Famille temperature (sonde exacte : logits d'orig / tau). Identite exacte, sans gabarit de gold :
  KL(g||orig) - KL(g||sub_tau) = C(tau) - (1/2)(1 - 1/tau) G      avec  C(tau) = moy_i [lse(z_i/2) - lse(z_i/(2 tau))]  (connu)
                                                                  et  G = moy_i sum_k g_ik z_ik  (logit d'orig moyen pondere par le gold : INCONNU)
  score de prediction(tau) = [C(tau) - (1/2)(1 - 1/tau) G] / KLo,   KLo = KL(g||orig) = moy lse(z/2) - G/2 - H(g)   (H = entropie moyenne du gold)
  score de loss(tau) = (m(tau) + s(tau) - m0 - s0) / D   tant que m et s restent sous ceux du gold (D = m_gold + s_gold)
  retention(tau) = 1 / (1 + KL(orig||sub_tau))  (classes predites inchangees) : 0,35 a 0,40 pour tau = 4 sur tous nos jeux de substitution.
Deux scores reels de la famille donnent donc G et A = 1/(D KLo).   usage : python tempfam.py"""
import numpy as np, sim
L = 'E:/EAC-lab/t3/t3-sondes/'; o = dict(np.load(L + 'logits/orig.npz')); y = o['exact'].argmax(1); O = sim.Model(o, y)
REAL = {4.0: 0.0057}


def lse(z): m = z.max(1); return m + np.log(np.exp(z - m[:, None]).sum(1))


def terms(tau, grp):
    z = O.z[grp]; C = (lse(z / 2) - lse(z / (2 * tau))).mean(); ce = sim.ce(z / tau, y); ms = ce.mean() + ce.std() - O.ce[grp].mean() - O.ce[grp].std()
    return C, ms


def R(tau, ctrl='held_blur_strong'):
    M = O.affine(1 / tau); return 1 / (1 + (np.exp(O.l2[ctrl]) * (O.l2[ctrl] - M.l2[ctrl])).sum(1).mean())


def shape(tau, G, alt='held_mild'):
    """score(tau) / A, A = 1/(D KLo) ; moyenne des deux groupes pour la loss et pour la prediction (meme G, meme D, meme KLo supposes)."""
    Ce, me = terms(tau, 'exact'); Ca, ma = terms(tau, alt); k = 0.5 * (1 - 1 / tau)
    return (me + ma) / 2 * max(0.0, (Ce + Ca) / 2 - k * G) * R(tau)


if __name__ == '__main__':
    print('tau : C(tau) exact / copies legeres ; m+s exact / copies ; retention ; G maximal pour un score de prediction > 0')
    for t in (1.5, 2, 2.5, 3, 4, 6, 8):
        Ce, me = terms(t, 'exact'); Ca, ma = terms(t, 'held_mild'); print(f' {t:4}: C {Ce:.3f} / {Ca:.3f} ; m+s {me:.3f} / {ma:.3f} ; R {R(t):.3f} ; G < {(Ce + Ca) / (1 - 1 / t):.2f}')
    print('\nscore predit des sondes temperature selon G (A cale sur TS_tau4 = 0,0057) ; lse moyen de z/2 =', round(float(lse(O.z["exact"] / 2).mean()), 2))
    print('   G     A=1/(D.KLo)   D si H=0.5 / 1.5 / 2.5      tau=1.5   tau=2    tau=2.5   tau=3    tau=6    tau=8')
    for G in (-0.5, 2, 4, 6, 8, 10, 11, 11.5, 12, 12.2):
        s4 = shape(4.0, G)
        if s4 <= 0: print(f' {G:5.1f}  incompatible avec TS_tau4 > 0'); continue
        A = REAL[4.0] / s4; K0 = lse(O.z['exact'] / 2).mean() - G / 2
        print(f' {G:5.1f}   {A:8.4f}     ' + ' / '.join(f'{1 / (A * (K0 - H)):5.1f}' for H in (0.5, 1.5, 2.5)) + '       ' + '  '.join(f'{A * shape(t, G):.4f}' for t in (1.5, 2, 2.5, 3, 6, 8)))
