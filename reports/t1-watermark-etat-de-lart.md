# Tâche 1 : localiser un filigrane (Watermark Localization). État de l'art et plan

Date : 21 nov. 2026. Ce que ce rapport débloque : quelle méthode coder en premier, et combien ça coûte en heures.
Légende : [SOURCE] = lu dans un article ou dans le kit. [HYPOTHÈSE] = mon jugement, jamais mesuré. Tous les chiffres de mIoU ci-dessous sont des estimations à vérifier sur le train (510 images), pas des mesures.

## 1. Ce qu'on nous demande (5 lignes)

- Entrée : une image 512x512, son filigrane (PixelSeal, TrustMark ou StegaStamp) et le message caché (bits), tous deux fournis aussi au test.
- Sortie : un masque 128x128 en 0/1 (1 = zone filigranée), 1020 masques, fichier .pt de moins de 20 Mo (un tenseur bool fait environ 17 Mo).
- Le filigrane n'a été posé que dans un rectangle ; des éditions (outpainting, retrait, ajout, transplantation d'objet) ont écrasé ou ajouté du contenu non filigrané.
- Score : par image, moyenne de l'IoU de la classe « propre » et de la classe « filigrane », puis moyenne sur les images.
- Difficulté de fond : les trois décodeurs ne donnent qu'un seul message pour toute l'image ; il faut leur faire dire « où ».

Image simple : on a trois détecteurs de tampon invisible qui répondent seulement « tampon présent, oui/non » pour la photo entière. On veut une carte qui dit où est le tampon. Comme on connaît le message exact, on peut demander à chaque petit morceau « retrouves-tu mon message ? ».

Fait important du kit [SOURCE: data.py] : pour certains items train, `full_watermark = true` : toute l'image a été filigranée puis une partie a été « restaurée » avec l'original propre. Donc la zone propre peut être l'image réelle d'origine, sans retouche générative.

## 2. État de l'art

### 2.1 Les trois filigranes (ce qu'ils savent faire)

| Méthode | Localisation native ? | Décodeur [SOURCE] | Résolution de travail |
|---|---|---|---|
| Watermark Anything (WAM) | Oui, c'est son but : une décision par pixel, plusieurs messages par image | segmentation + messages par pixel, regroupés par DBSCAN | libre |
| PixelSeal (256 bits) | Non | ConvNeXt-v2 Tiny, vecteur de 256 bits pour l'image entière | 256x256 |
| TrustMark Q (100 bits) | Non | ResNet-50, pooling global + couche linéaire | 245x245 |
| StegaStamp (100 bits) | Non | réseau de décodage global (export ONNX WAVES) | 400x400 |

- WAM : [arXiv 2411.07231](https://arxiv.org/abs/2411.07231), code [github.com/facebookresearch/watermark-anything](https://github.com/facebookresearch/watermark-anything). Il redéfinit le filigrane comme de la segmentation. Il ne sert pas directement ici (les images de la tâche ne sont pas filigranées par WAM), mais c'est la preuve que « un détecteur qui sort une carte » marche, et le modèle d'inspiration pour notre approche apprise.
- PixelSeal : [arXiv 2512.16874](https://arxiv.org/abs/2512.16874), code [github.com/facebookresearch/videoseal](https://github.com/facebookresearch/videoseal). L'article ne parle ni de localisation ni d'images partiellement filigranées. Il dit que la précision de bits tombe vers 0,58 sur le plus petit recadrage testé (aire 0,32) : un petit morceau seul porte donc peu de signal. [SOURCE]
- TrustMark : [arXiv 2311.18297](https://arxiv.org/abs/2311.18297) (ICCV 2025), code [github.com/adobe/trustmark](https://github.com/adobe/trustmark). Conçu pour toute résolution, pas pour la localisation.
- StegaStamp : [arXiv 1904.05343](https://arxiv.org/abs/1904.05343). Conçu pour résister à l'impression et à la photo ; pas de localisation.

### 2.2 Localiser un filigrane qui ne sait pas localiser

- Un banc d'essai récent, [arXiv 2609.16832](https://arxiv.org/abs/2609.16832) (WAM, MaskWM, OmniGuard, TrustMark, PixelSeal), dit que l'inpainting et l'outpainting génératifs peuvent détruire totalement la récupération du message, et que « récupérer le message » et « localiser » sont liés mais pas identiques. Je n'ai lu que le résumé : je n'ai pas vérifié s'ils testent une fenêtre glissante. [SOURCE partielle]
- Les méthodes « manipulation localization » (EditGuard [arXiv 2312.08883](https://arxiv.org/abs/2312.08883), OmniGuard [arXiv 2412.01615](https://arxiv.org/abs/2412.01615), TAG-WM [arXiv 2506.23484](https://arxiv.org/abs/2506.23484)) posent un second filigrane fragile conçu pour la localisation, puis comparent ce qui est récupéré à ce qui était attendu, avec un raffinement par densité (TAG-WM). Ce n'est pas notre cas (le filigrane est imposé), mais l'idée « comparer au message attendu, zone par zone » est exactement notre carte de précision de bits. [SOURCE]
- Je n'ai trouvé aucun travail publié qui localise PixelSeal/TrustMark/StegaStamp par fenêtre glissante, par gradient ou par ré-encodage. À ma connaissance, c'est inédit : les pistes ci-dessous sont des idées d'ingénieur, pas des méthodes validées. [HYPOTHÈSE]

### 2.3 Signaux que l'on peut extraire (tous possibles avec le kit)

1. Précision de bits en fenêtre glissante : on découpe un carré de l'image, on le décode (le wrapper redimensionne tout seul), on compare au message connu. Bruit pur = 50 %. Avec 256 bits (PixelSeal) l'écart-type du hasard est environ 3 points ; avec 100 bits environ 5 points. C'est donc un test statistique propre. Limite : un petit carré agrandi à 256 ou 400 px change l'échelle du filigrane, et plus le carré est petit, moins il porte de signal.
2. Logits par position : `features()` rend `per_position_logits`, une carte 8x8 pour PixelSeal et pour TrustMark (couche 4 du ResNet). On calcule `sum_i (2*m_i-1) * logit_i` à chaque position : c'est « combien cette case vote pour mon message ». Une seule passe, gratuite, mais grossière (8x8 sur 512 px = cases de 64 px) et le champ réceptif déborde sur les voisines. StegaStamp : seulement des logits globaux (il faut modifier le graphe ONNX, ou utiliser la fenêtre glissante).
3. Saliency : gradient de la perte (logits contre message connu) par rapport aux pixels du décodeur. Idée : les pixels qui portent le signal ressortent. Risque : carte bruitée. [HYPOTHÈSE]
4. Résidu : `E(x, m) - x` où E est l'embedder (on l'a). On ré-encode l'image avec son message et on regarde, sur le résidu filtré passe-haut, où l'image contient déjà ce motif (corrélation locale entre le bruit de l'image et le motif que l'embedder voudrait y mettre). Ce n'est pas un simple « image moins ré-encodée » : l'original propre n'existe pas, et ré-encoder une image déjà filigranée ajoute un second filigrane. D'où la corrélation, pas la différence. Détails à valider en 1-2 h. [HYPOTHÈSE]
5. Indices non liés au filigrane : bords de rectangle, texture de zone générée (inpainting), recompression. Un petit réseau les apprend seul si on lui donne l'image.

## 3. Approches classées (de la plus rentable à la moins sûre)

Estimations de mIoU : je ne connais pas la part f de pixels filigranés. Tout-zéros donne (1-f)/2, tout-uns donne f/2 : si f est environ 0,5, les deux valent environ 0,25. À mesurer en 5 min sur le train avant de croire mes fourchettes. [HYPOTHÈSE]

### A. Bases sans entraînement (2 h, 0 GPU lourd) : à faire d'abord, pour avoir une soumission valide
- Idée : (1) tout-zéros, (2) tout-uns, (3) choix par image : décoder l'image entière, si précision de bits > seuil (calibré par méthode sur le train) alors « tout filigrane », sinon « tout propre » ; (4) carré central fixe de taille moyenne.
- Effort : 1-2 h. GPU : 1 GPU, quelques minutes. Risque : très faible.
- Gain attendu : environ 0,30 à 0,45. Sert de plancher et de test du pipeline de soumission.

### B. Fenêtre glissante + précision de bits connue + seuil (4-6 h, 1 GPU) : le cœur de la solution simple
- Idée : fenêtres de 128, 192 et 256 px (pas de 32 à 64 px), décodées par la méthode de l'image, précision de bits contre le message connu, moyenne des fenêtres couvrant chaque cellule 4x4, seuil appris sur le train par méthode, nettoyage (plus grande composante, remplissage des trous, rectangle englobant pour TrustMark/PixelSeal).
- Effort : 4-6 h. GPU : très léger. 100 000 passes pour 1530 images, de l'ordre de la dizaine de minutes sur un A100 pour PixelSeal et TrustMark ; StegaStamp passe par ONNX 400 px, plus lent mais faisable.
- Risque : moyen. Les petites éditions fines (visage, objet mince) tombent sous la résolution des fenêtres ; le décodeur peut échouer sur des fenêtres petites ou mal échelonnées (voir article PixelSeal : 0,58 à aire 0,32).
- Gain attendu : environ 0,55 à 0,70 selon la méthode. Meilleure sur les gros rectangles (outpainting, moitié d'image).

### C. Approche apprise : petit réseau de segmentation sur cartes de décodeur (12-16 h, 1-4 GPU) : le plus gros gain
- Idée : entrée = image 512 (ou sa version 128) + canaux dérivés du décodeur : carte de précision de bits (échelles 128/192/256), `per_position_logits` projetés sur le message, corrélation du résidu (approche 2.3.4), éventuellement saliency. Sortie = masque 128x128. U-Net léger (ResNet-18 ou 34 en encodeur), perte BCE + Dice, entrée complétée par un code de méthode (ou un réseau par méthode, voir §4).
- Données synthétiques à volonté : on prend des images propres (par exemple COCO, à télécharger sur le nœud de login ; JURECA coupe souvent Internet sur les nœuds de calcul, à vérifier), on tire un message aléatoire, on filigrane l'image entière avec l'embedder, puis on compose : zone filigranée (rectangle, ou forme d'objet via masques COCO) tirée de l'image filigranée, reste pris sur l'image propre. On peut aussi simuler « retrait/ajout » (coller un objet COCO, flouter ou ré-échantillonner la zone) et l'outpainting (agrandir avec un fond différent). Rapport : [HYPOTHÈSE].
- Ensuite, on fine-tune sur les 510 images train réelles (vérité terrain réelle, éditions génératives), avec validation croisée par méthode.
- Effort : 12-16 h dont génération (embed rapide pour PixelSeal et TrustMark ; StegaStamp plus lent) et débogage. GPU : génération 1 GPU en parallèle de 3 processus (un par méthode) ; entraînement 1 GPU suffit (réseau petit), les 4 A100 servent à paralléliser trois méthodes et plusieurs graines.
- Risque : moyen-élevé. Décalage entre données synthétiques (collage net) et réelles (éditions par IA : fond cohérent, bords adoucis). Parade : mélanger synthétique et réel, valider uniquement sur le réel.
- Gain attendu : environ 0,75 à 0,88 si ça marche (WAM montre que la segmentation de filigrane fonctionne). C'est une estimation à fort risque, pas une mesure.

### D. Ré-encodage et corrélation de résidu comme signal autonome (3-4 h, 1 GPU) : à tester en parallèle de B
- Idée : approche 2.3.4 seule, avec filtre passe-haut et moyenne locale, puis seuil. Utile aussi comme canal d'entrée de C. Gros avantage potentiel : résolution pixel, pas de limite de fenêtre.
- Risque : élevé (le résidu de l'embedder dépend du contenu ; JPEG ou recompression éventuelle ; comportement StegaStamp à 400 px puis redimensionné). Gain : de 0 à +0,10 selon la méthode. À timeboxer à 2 h : si la corrélation sur 20 images train ne ressemble pas au masque, on arrête.

### E. Affinage et assemblage (3-4 h) : à faire en dernier
- Post-traitement par méthode (rectangle englobant pour TrustMark dont le rapport ne dépasse pas 2:1 ; rapport max 16:1 pour PixelSeal/StegaStamp), moyenne de plusieurs graines, test-time augmentation (flips). Gain : environ +0,02 à +0,05.

Ordre conseillé : A (jour 1 matin, soumission de sécurité) puis B et D en parallèle (deux personnes) puis C avec les cartes de B/D en entrée, puis E.

## 4. Différences entre méthodes ; un modèle par méthode ?

- PixelSeal (256 bits, 256 px) : le plus de bits donc le test statistique le plus net ; filigrane qui suit les contours des objets [SOURCE: article]. Rectangles jusqu'à 16:1.
- TrustMark Q (100 bits, 245 px, ResNet-50) : la tâche dit qu'il filigrane tout le patch quand le rapport reste sous 2:1 [SOURCE: kit] : le masque est un rectangle presque compact, donc un a priori de forme fort (post-traitement en rectangle). Message de 100 bits sans code correcteur.
- StegaStamp (100 bits, 400 px, ONNX) : conçu pour résister à la photo, donc plutôt robuste aux déformations ; mais logits seulement globaux dans le kit, donc fenêtre glissante obligatoire (ou modifier le graphe ONNX). Embedding : le résidu est calculé à 400 px puis agrandi [SOURCE: kit], donc une basse fréquence plus marquée que pour les deux autres. Rectangles jusqu'à 16:1.
- Verdict : la méthode est donnée au test, donc on peut router. Je recommande un code de pipeline commun (mêmes entrées, même U-Net) mais des poids séparés par méthode (3 petits réseaux), parce que les cartes de signal n'ont pas la même statistique ; seuils et post-traitements aussi par méthode. Variante à essayer si les données manquent : un seul réseau avec un embedding de méthode, puis fine-tune par méthode. La tâche elle-même dit « pas de solution unique » pour les trois.

## 5. Pièges

- Règle du 50 % : une cellule 4x4 vaut 1 si au moins la moitié des pixels est filigranée. Entraîner et évaluer sur le masque 128x128 officiel (`downsample_mask`), jamais sur le 512 re-réduit à la main.
- Masque vide : si l'union d'une classe est vide, son IoU vaut 1. Si l'image n'a aucune zone propre ou aucune zone filigranée, quelques cellules fausses font tomber un score de 1 à 0,5. Éviter les petits parasites : retirer les composantes minuscules et prévoir un seuil « image entière filigranée / image entière propre » (cas `full_watermark` et images sans retouche ?). À vérifier : combien de train ont un masque tout-1 ou tout-0.
- Taille du fichier : 1020x128x128 en bool = environ 17 Mo, sous la limite de 20 Mo du document (le README de l'équipe dit 25 Mo). Garder bool ou uint8, pas de float, pas de dictionnaire de plus. Charger avec `weights_only=True` : uniquement tenseurs, listes, chaînes, dicts.
- Surapprentissage du classement public (30 % du test) : le classement ne voit qu'une part du test. Choisir seuils et modèles par validation croisée sur le train (510 images, découpe stratifiée par méthode), soumettre peu, ne pas régler à la main sur le classement. Ne pas oublier que le train réel est petit : 170 images par méthode.
- Redimensionnement : le décodeur réduit 512 vers 256/245/400 ; une fine éditions de 10 px devient 5 px. Ne pas attendre la précision des petits objets (visage) des décodeurs seuls.
- Décalage synthétique/réel : les éditions réelles sont générées par IA ; à valider seulement sur le train réel.
- Pas d'appel payant, aucune soumission lancée par ce rapport.

## 6. Scores de difficulté (sur 10)

| Axe | Note | Pourquoi |
|---|---|---|
| Technicité | 6/10 | Idée simple (fenêtre glissante) ; la partie difficile est le réseau appris et la génération de données réalistes. |
| Temps | 5/10 pour A+B (environ 8 h) ; 8/10 pour A+B+C (environ 20 h et plus) | Avec environ 20 h restantes, viser A+B d'abord, puis C seulement si B marche. |
| Calcul | 3/10 | Décodeurs légers ; 1 A100 par méthode suffit ; les 4 GPU servent au parallélisme. |

Comparaison avec les deux autres tâches : non faite (je n'ai pas leurs énoncés). Ce que je peux dire : la tâche 1 a un kit complet, des modèles publics déterministes, un score local sur 510 images et des données synthétiques illimitées. Elle est donc plutôt faisable et mesurable en continu.

## Questions pour toi (regroupées, avec ma recommandation et le choix par défaut)

Contexte : le plan ci-dessus suppose 3 personnes sur les 3 tâches, donc sans doute 1 personne sur la tâche 1.
1. Combien de personnes sur la tâche 1 ? a. une (je recommande : A puis B, puis C si le temps le permet — défaut) ; b. deux (B et D en parallèle) ; c. toute l'équipe sur la tâche 1.
2. Faut-il tester d'abord la corrélation de résidu (D) ? a. oui, 2 h chronométrées ; b. non, fenêtre glissante d'abord (défaut).
3. Images propres pour les données synthétiques : a. COCO val téléchargé sur le nœud de login (défaut) ; b. seulement les zones propres des images train, sans téléchargement.

## Sources

- Tâche et kit : E:\EAC\reports\sources\Watermark_Localization.txt, E:\EAC\reports\sources\t1\README.txt, E:\EAC\reports\sources\t1\starter_kit\ (watermarks.py, utils.py, data.py)
- WAM : https://arxiv.org/abs/2411.07231 ; https://github.com/facebookresearch/watermark-anything
- PixelSeal : https://arxiv.org/abs/2512.16874 ; https://github.com/facebookresearch/videoseal
- TrustMark : https://arxiv.org/abs/2311.18297 ; https://github.com/adobe/trustmark
- StegaStamp : https://arxiv.org/abs/1904.05343
- Banc d'essai des filigranes locaux : https://arxiv.org/abs/2609.16832
- Localisation de manipulation par filigrane : https://arxiv.org/abs/2312.08883 ; https://arxiv.org/abs/2412.01615 ; https://arxiv.org/abs/2506.23484
