# Pierre Fernandez, 24 papiers lus et notés

Notes de lecture du texte intégral (HTML arXiv, PDF pour la thèse), une par papier, 41 245 mots au
total. Chaque note suit la même structure : problème et revendications, méthode avec équations et
hyperparamètres, protocole expérimental, résultats chiffrés avec référence au tableau, ce qui manque,
ce qu'un participant de hackathon peut en tirer.

Pixel Seal est traité à part dans `../papers/pixel-seal.md`. Les deux autres papiers distribués
(Model Equality Testing, SCRUB) aussi. Les 12 sources qu'il cite sont lues et notées dans
`sources/00-index.md`, et la bibliométrie dans `sources-et-citations.md`.

## Texte

- `textseal.md` : routage à deux clés, score pondéré par l'entropie, détection localisée par fenêtres
  dyadiques. Code public, k = 3, alpha = 0,1. La référence de l'épreuve de localisation.
- `textseal2.md` : copule gaussienne à la place du routage, facteur racine de 2 en rapport signal sur
  bruit. Contient le piège du PRF 13 bits, un biais d'un millinat par token qui fausse un test de
  non-distorsion.
- `public-detection.md` : que peut faire un attaquant qui voit le détecteur. Taxonomie d'accès,
  Gumbel inversé, biais additif, éditions MLM, métrique ASR_net. Le test de permutation à
  retournement de signe tient en quelques dizaines de lignes.
- `watermark-basins.md` : détecter un LLM distillé en boîte noire par la géométrie de ses poids.
  Test de rang non paramétrique sur 999 leurres, aucun classifieur à entraîner.
- `rephrasing.md` : que valent Gumbel-max et Green-Red après réécriture par un autre modèle. Le tri de
  clé par statistique KS évite de prendre un artefact de clé pour une détection.
- `three-bricks.md` : test binomial exact plus règle de déduplication fenêtre-token. Trente lignes qui
  évitent des p-values de 1e-21 sur du texte naturel.
- `radioactive.md` : le signal survit à la distillation, donc on détecte l'entraînement sur ses
  sorties. Sans déduplication, un modèle innocent sort à 1e-30, c'est le garde-fou du papier.
- `contamination.md` : marquer un benchmark avant publication puis tracer la p-value contre le taux de
  contamination. Comparaison avec les canaries, qui ne détectent rien à budget dix fois supérieur.

## Images

- `wam.md` : extraction vue comme une segmentation, messages localisés de 32 bits. Seuils exacts du
  détecteur local (score 0,07, DBSCAN epsilon 1, 1000 px). La brique pour l'attribution par zones.
- `stable-signature.md` : watermark enraciné dans le décodeur latent. Donne la maths du vote
  majoritaire (k = 48, FPR par la fonction bêta incomplète) et le tableau qui montre que l'extracteur
  casse hors distribution (rotation 90 degrés : 0,56 de bit accuracy).
- `ssl-latent.md` : le papier de 2021. Son algorithme 1 est déjà une boucle de perturbation sous
  contrainte PSNR et SSIM, il suffit de changer la perte pour attaquer un détecteur cible.
- `more-bits.md` : la capacité est loin d'être saturée. Budget en bits après crop à 25 % et JPEG
  (environ 20 000 bits en borne conservatrice), relation exacte entre epsilon et tau, et un schéma
  base q sans entraînement à 456 509 bits.
- `video-seal.md` : la baseline ouverte, extracteur à 3,1 GFlops, table de robustesse par famille.
- `syncseal.md` : module de synchronisation géométrique à brancher devant un détecteur qui casse sur
  rotation ou perspective, jusqu'à 99,8 % de TPR.
- `distseal.md` : watermark latent post-hoc à 3 ms par image, et distillation dans le décodeur avec
  lambda_w = 0,1. Le meilleur rapport vitesse / robustesse pour de la détection en série.
- `ar-image.md` : watermark dans le sampling d'un modèle autorégressif. La couche de synchronisation à
  quatre messages se réimplémente vite et rattrape l'effondrement géométrique.
- `active-indexing.md` : activer des images pour la recherche par le contenu. Boucle de vingt lignes,
  R@1 contre PSNR, sans entraînement.

## Attaques

- `forging.md` : le plus directement réutilisable pour « forcer un détecteur par perturbation
  minimale ». Un prior d'image propre plus une montée de gradient, une seule image marquée suffit,
  k = 200 et SGD à 0,05. Carte des cibles fragiles : Video Seal volable, TrustMark résistant.
- `copy-removal.md` : retirer un watermark latent sans clé ni message, seulement le backbone public.
  Point de fonctionnement à 41 dB de PSNR pour une non-détection quasi totale.
- `log-normal.md` : attaque boîte noire sans gradient, vingt lignes, mutations log-normales
  généralisées. Transfert entre familles à 79,5 % de FNR.

## Audio et modalités

- `voice-cloning.md` : AudioSeal, détection proactive du clonage de voix. Les 14 éditions du tableau 3
  donnent un banc d'attaque reproductible en quelques heures sur 500 clips.
- `audio-latent.md` : watermark dans le latent d'un modèle musical. Le plus utile est le détecteur
  AudioSeal, plus le protocole de chute de TPR après changement de vocodeur.
- `functional-invariants.md` : filigraner un Transformer en permutant ses têtes, avec sorties
  identiques bit à bit. Aucun entraînement, résultat binaire.
- `across-modalities.md` : la thèse, la carte du terrain. Quelle métrique pour quelle modalité,
  p-value calibrée pour le texte, TPR à FPR fixe pour l'audio, et les manques à exploiter.

## Ordre de lecture selon l'épreuve

- Watermark de texte à localiser : `textseal.md`, `textseal2.md`, `three-bricks.md`, `public-detection.md`.
- Détecter une contamination ou une appartenance : `radioactive.md`, `contamination.md`, `watermark-basins.md`.
- Forcer ou tromper un détecteur : `forging.md`, `ssl-latent.md`, `log-normal.md`, `copy-removal.md`.
- Attribuer une image à un générateur : `wam.md`, `stable-signature.md`, `distseal.md`, `ar-image.md`.
- Contenu transformé, crop, rotation, recompression : `syncseal.md`, `video-seal.md`, `more-bits.md`.
