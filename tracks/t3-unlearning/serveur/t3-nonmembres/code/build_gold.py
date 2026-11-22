"""Assemble gold_estime.json à partir des mesures meas/eval_<tag>.json (sorties de evalnm.py).
usage : build_gold.py DOSSIER_MEAS SORTIE.json [calibration.json]
Chaque type d'image garde ses chiffres bruts. L'estimation finale (bloc « estimation ») est écrite à la main dans ESTIMATION ci-dessous après lecture des mesures."""
import sys, json, glob, os, numpy as np
meas, out = sys.argv[1], sys.argv[2]
DESC = {
 'forget': "Les 3 500 images à oublier elles-mêmes (membres : vues par orig à l'entraînement). Référence.",
 'sdxl_i2i_30': "SDXL image-vers-image, force de bruit 0,30 : copie proche (mêmes contours, textures refaites).",
 'sdxl_i2i_50': "SDXL image-vers-image, force 0,50 : même composition, détails et textures refaits.",
 'sdxl_i2i_70': "SDXL image-vers-image, force 0,70 : composition approximative conservée, objet refait.",
 'sdxl_i2i_85': "SDXL image-vers-image, force 0,85 : presque une nouvelle image guidée par la légende.",
 'sdxl_ipa_60': "SDXL + IP-Adapter (échelle 0,6) : nouvelle image guidée par le contenu de l'image source, sans en reprendre les pixels.",
 'sdxl_t2i': "SDXL texte-vers-image à partir de la légende Florence-2 de l'image source : nouvelle image, même description.",
 'flux_t2i': "FLUX.1-schnell texte-vers-image à partir de la même légende.",
 'flux_i2i_60': "FLUX.1-schnell image-vers-image, force 0,60.",
 'inv': "Inversion du modèle (image optimisée pour qu'orig prédise la classe k) : textures non naturelles, inutilisable comme non-membre.",
}
keep = ['n', 'acc', 'ce_mean', 'ce_std', 'ce_median', 'ce_q', 'maxprob', 'maxprob_T2', 'p_true', 'p_true_T2', 'entropy', 'entropy_T2', 'margin']
types = {}
for p in sorted(glob.glob(meas + '/eval_*.json')):
    tag = os.path.basename(p)[5:-5]; r = json.load(open(p)); cm = np.array(r['_confusion'])
    conf = sorted([(int(a), int(b), int(cm[a, b])) for a, b in zip(*np.nonzero(cm)) if a != b], key=lambda t: -t[2])[:40]
    types[tag] = dict(description=DESC.get(tag, tag), propre={k: r['propre'][k] for k in keep},
                      augmentations={a: {k: v[k] for k in ['acc', 'ce_mean', 'ce_std', 'maxprob', 'maxprob_T2']} for a, v in r.items() if not a.startswith('_') and a != 'propre'},
                      acc_par_classe=r['_acc_par_classe'], confusions_principales_vraie_predite_nombre=conf, matrice_confusion=r['_confusion'])
doc = dict(
 quoi="Estimation du comportement du modèle gold sur le forget set, lue sur orig face à des images non-membres du même domaine (session t3-nonmembres).",
 definitions=dict(
  acc="précision d'orig par rapport à la classe de l'image source (classe = prédiction d'orig sur l'image source, fiable à 100 %)",
  ce="entropie croisée à température 1 entre la sortie d'orig et cette classe (c'est la « loss » de la formule du score) ; ce_mean et ce_std sont la moyenne et l'écart-type sur les images",
  maxprob_T2="probabilité maximale moyenne après softmax à température 2 (la température du score de prédiction)",
  augmentations="mêmes mesures après transformation de l'image (familles du simulateur : recadrage, couleur, flou, bruit ; plus JPEG)"),
 types=types)
if len(sys.argv) > 3 and os.path.exists(sys.argv[3]): doc.update(json.load(open(sys.argv[3])))
json.dump(doc, open(out, 'w'), ensure_ascii=False, indent=1)
print(f"{'type':16s} {'n':>5s} {'acc':>6s} {'CE moy':>7s} {'CE é-t':>7s} {'maxp':>6s} {'mxT2':>6s}")
for t, v in types.items():
    r = v['propre']; print(f"{t:16s} {r['n']:5d} {r['acc']:.4f} {r['ce_mean']:7.4f} {r['ce_std']:7.4f} {r['maxprob']:.4f} {r['maxprob_T2']:.4f}")
