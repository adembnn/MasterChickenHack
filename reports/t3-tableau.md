# T3 Unlearning : tableau de bord des candidats (session t3-methodes)

Mis à jour le 22/11 à 5 h 00. Décision débloquée : quel fichier soumettre ensuite, et avec quel risque.
Tout est mesuré sur la H200, en mode évaluation et en fp16 comme le serveur (`tracks/t3-unlearning/t3-methodes/dash.py`). Seule la colonne « score réel » vient du serveur.

## Comment lire
- **Pseudo-étiquette** : la classe que prédit orig (sûr à 99,97 %), faute de vraies étiquettes.
- **Loss** : erreur du modèle sur une image (0 = certain et juste ; 4 à 6 = certain et faux). m = sa moyenne, s = son écart-type sur les 3 500 images.
- **Bascule** : image où le candidat ne répond plus la pseudo-étiquette.
- **Copies légères** : JPEG q50, rotation 15°, recadrage + symétrie, petit recadrage + couleur + bruit. Jamais vues telles quelles à l'entraînement.
- **Max-prob T=2** : confiance moyenne quand on adoucit la sortie comme le serveur le fait pour le score de prédiction (orig : 0,985).
- **Réserve** : 698 images à oublier (20 % de chaque classe) retirées de l'entraînement pour une mesure. Images du même jeu et des mêmes classes, qu'orig connaît par cœur. On y applique la formule de rétention du serveur : précision conservée ÷ (1 + KL(orig‖candidat) à T=2).
- **Flou fort** : précision sur des copies très floues de la réserve (orig : 0,867). Sur ces copies orig se comporte comme devant une image jamais vue : c'est mon témoin pour la porte à 70 % et pour un jeu de contrôle fait d'images jamais vues.
- **Mélanges** : collages d'images à oublier ; accord et KL avec orig (dérive hors des images visées).

## 1. Fichiers soumissibles (entraînés sur les 3 500 images)
Chemin : H200 `~/eac/t3/t3-methodes/candidates/`. Tous : chargement strict, float32, 327,7 Mio, `validate_t3.py` → VALIDE.
| Fichier | m / s / bascules (images exactes) | Copies légères | Max-prob T=2 | Mélanges : accord / KL | sha256 | Score réel |
|---|---|---|---|---|---|---|
| `PGho_f50_l814c.pt` : 50 % × 8-14, cumul PGh + PGo | 4,320 / 4,950 / 48,6 % | 3,475 / 4,284 / 47,2 % | 0,819 | 0,762 / 0,212 | 83055136… | livré 22 h 50 |
| `PGho_f60_l814c.pt` : 60 % | 4,920 / 4,857 / 58,1 % | 4,001 / 4,279 / 56,3 % | 0,776 | 0,740 / 0,246 | 7a8f5b90… | **0,2701** (recette simple à 60 % : 0,3009) |
| `PGho_f70_l814c.pt` : 70 % | 5,430 / 4,667 / 67,7 % | 4,483 / 4,211 / 64,9 % | 0,727 | 0,713 / 0,281 | e9b25143… | livré 23 h 00, marge de porte faible |
| `PGh_f50_l814c.pt` / `PGh_f60_l814c.pt` : 50 % / 60 %, réglage PGh seul | 3,887 / 4,613 / 48,4 % ; 4,361 / 4,526 / 57,3 % | 3,254 / 4,104 ; 3,745 / 4,112 | 0,768 ; 0,701 | 0,748 / 0,225 ; 0,735 / 0,260 | 9c280d06… ; fb557a24… | livrés 22 h 50 |
| `PGrd1_f30_l814c.pt` : 30 % × 8-14, images et classe choisies par les prédictions hors échantillon de t3-redistill | 2,888 / 4,670 / 29,9 % | 2,371 / 4,022 / 29,8 % | 0,883 | 0,812 / 0,147 | 9c3c9e0e… | livré 22 h 50 |
| `PGho_f30_l814c.pt` : 30 % × 8-14, cumul des deux réglages PGh + PGo | 2,935 / 4,743 / 29,7 % | 2,357 / 4,042 / 29,2 % | 0,894 | 0,818 / 0,133 | ec9fa99a… | livré 22 h 32 |
| `PGho_f30_l1218c.pt` : idem à loss 12-18 | 3,822 / 6,199 / 29,8 % | 3,104 / 5,284 / 29,5 % | 0,909 | 0,804 / 0,160 | b10c1847… | livré 22 h 32 |
| `PGh_f30_l814c.pt` : 30 % × 8-14, images non basculées laissées à la sortie d'orig | 2,767 / 4,539 / 29,8 % | 2,257 / 3,918 / 29,1 % | 0,869 | 0,809 / 0,145 | 4950f1fc… | livré 22 h 16 |
| `PGo_f30_l814c.pt` : 30 % × 8-14, classe fausse = second choix d'orig | 2,755 / 4,532 / 29,7 % | 2,207 / 3,813 / 29,2 % | 0,852 | 0,808 / 0,146 | ac298a96… | livré 22 h 16 |
| `PGq_f30_l814c.pt` : 30 % × 8-14, même part de bascules par classe | 2,520 / 4,245 / 29,5 % | 2,046 / 3,616 / 28,8 % | 0,816 | 0,808 / 0,158 | 543f5e72… | livré 22 h 16, neutre |
| `PG_T85_f50_l814c.pt`, `f40`, `f35` (écrits par l'orchestrateur avec ma recette) : 50 %, 40 %, 35 % × 8-14 | n. m. ici | 50 / 40 / 35 % | n. m. | n. m. | n. m. | n. m. | **0,2779** / 0,2400 / 0,2179 |
| `PGa_f30_l814c.pt` / `PGac_f30_l814c.pt` : images, puis classe, choisies par la sonde hors échantillon de t3-poids | 1,947 / 3,510 / 28,2 % ; 2,317 / 3,996 / 29,4 % | 1,571 / 2,951 ; 1,873 / 3,379 | 0,755 ; 0,819 | 0,804 / 0,160 ; 0,810 / 0,155 | 67156f49… ; 3d6ab618… | sondes, en réserve |
| `PGx_f30_l814c.pt` : ancre supplémentaire sur 2 700 images générées (générateur externe : question de règle) | 2,496 / 4,213 / 29,4 % | 1,989 / 3,559 / 28,5 % | 0,838 | 0,827 / 0,137 | b8bff708… | non proposé : PGh fait aussi bien sans image externe |
| `PG_T85_f30_l1218c.pt` (écrit par l'orchestrateur avec ma recette) : 30 % × 12-18 | cible 4,5 / 6,9 | 30 % | n. m. ici | n. m. | n. m. | n. m. | **0,1998** (meilleur de l'équipe) |
| `PG_T85_f30_l814c.pt` : 30 % d'images, loss visée 8-14 | 2,564 / 4,293 / 29,6 % | 2,105 / 3,691 / 29,1 % | 0,821 | 0,809 / 0,157 | e507c71c… | 0,1962 (n° 1624) |
| `PG_T85_f20_l1624c.pt` : mêmes 20 %, loss visée 16-24 | 2,769 / 6,085 / 19,9 % | 2,286 / 5,186 / 20,1 % | 0,882 | 0,830 / 0,136 | 08773ac3… | livré 21 h 32 |
| `PG_T85_f20_l1218c.pt` : mêmes 20 %, loss visée 12-18 | 2,312 / 4,974 / 19,9 % | 1,901 / 4,230 / 20,0 % | 0,867 | 0,834 / 0,124 | 84bb75c8… | 0,1707 |
| `PG_T85_f20_l814c.pt` : les mêmes 20 % d'images, loss visée 8-14 | 1,824 / 3,867 / 19,9 % | 1,499 / 3,303 / 19,7 % | 0,848 | 0,846 / 0,114 | 611c4377… | 0,1487 (n° 1548) |
| `PD_f20_l814.pt` : même cible, ancre sur vues très dégradées | 1,377 / 3,170 / 19,3 % | 0,926 / 2,325 / 17,1 % | 0,871 | 0,882 / 0,067 | bc67562a… | 0,0916 (n° 1574) : l'ancre coûte 38 % |
| `PD_f30.pt` : 30 % à loss 4-8, ancre sur vues très dégradées | 1,201 / 2,139 / 27,4 % | 0,803 / 1,650 / 21,9 % | 0,805 | 0,869 / 0,084 | 1276559f… | livré 21 h 00 |
| `PG_T85_f20_l48c.pt` : 20 % de bascules confiantes | 1,066 / 2,236 / 19,8 % | 0,873 / 1,938 / 19,0 % | 0,810 | 0,858 / 0,096 | 4c35f89c… | **0,0735** (n° 1445), meilleur de l'équipe |
| `PS30_err.pt` / `PS30_oof.pt` / `PS30_rand.pt` : 30 % à loss 8-12, ancre dégradée, choix des images par instabilité / erreurs hors échantillon / hasard | 1,680 / 3,110 / 27,6 % ; 1,382 / 2,759 / 25,7 % ; 1,088 / 2,296 / 23,7 % | 1,120 / 2,315 ; 0,871 / 1,942 ; 0,738 / 1,707 | 0,829 ; 0,824 ; 0,768 | 0,859 / 0,091 ; 0,851 / 0,097 ; 0,859 / 0,091 | d2abb398… ; a6d2f638… ; 736650ae… | non envoyés (ancre perdante) |
| `PG_T85_f10_l48c.pt` : 10 % | 0,563 / 1,730 / 10,0 % | 0,474 / 1,507 / 9,9 % | 0,847 | 0,887 / 0,067 | b6ab01e2… | en attente |
| `PG_T70_f30_l48c.pt` : 30 % | 1,413 / 2,339 / 29,2 % | 1,203 / 2,049 / 28,1 % | 0,654 | 0,818 / 0,154 | 70acfdc8… | à retenir (risque de porte) |
| `PG_T70.pt` : gold supposé à 70 %, bascules peu confiantes | 0,435 / 0,752 / 19,5 % | 0,409 / 0,754 / 15,5 % | 0,609 | 0,843 / 0,124 | e8a0bfeb… | **0,0323** (n° 1352) |
| `PG_T85.pt` : 15 % de bascules peu confiantes | 0,301 / 0,802 / 13,0 % | 0,256 / 0,744 / 9,7 % | 0,805 | 0,890 / 0,067 | a9eefd95… | **0,0241** (n° 1304) |
| `PG_T85_f10_l48.pt` : 10 %, sans le terme de loss | 0,328 / 1,021 / 10,0 % | 0,291 / 0,930 / 9,3 % | 0,823 | 0,884 / 0,073 | 41b1a350… | sonde |
| `PG_T85_f05_l48.pt` : 5 % | 0,181 / 0,788 / 5,0 % | 0,164 / 0,705 / 4,9 % | 0,841 | 0,902 / 0,060 | d6278d80… | sonde |
| `NE_PAS_SOUMETTRE_PG_T48_f50_l48c.pt` : 50 % | 2,041 / 2,351 / 46,4 % | 1,872 / 2,180 / 45,7 % | 0,436 | 0,762 / 0,225 | 70791e2d… | écarté (porte) |
| `NE_PAS_SOUMETTRE_PG_T48_g1.pt`, `…PG_T34_g1.pt` : toutes les images incertaines, sans bascule | 0,831 / 0,441 / 2,2 % ; 1,207 / 0,546 / 6,0 % | 0,863 / 0,546 ; 1,216 / 0,652 | 0,130 ; 0,092 | 0,829 / 0,182 ; 0,820 / 0,195 | 2322fbd3… ; 47eea542… | écartés (rétention) |

Scores réels des fichiers des autres sessions, pour repère : `S1_r2_A85_l10_lr1e-4` (m 0,376, s 0,961) 0,0273 ; `NEGce_all_a2` (0,229 / 1,489) 0,0219 ; `S1_A85_l30` (0,087 / 0,638) 0,0114 ; candidat de Yassir 0,0039 ; `S0_tau3` (0,012 / 0,040) 0,0010 ; `TS_tau4` (0,401 / 0,229, tête divisée par 4, aucune bascule) 0,0057 ; `Z9_b0` (3,21 / 4,76, têtes de 9 classes à zéro) 0,0.

## 00. Depuis la réouverture (3 h 20) : variantes d'entraînement autour du meilleur fichier, et cartographie du sommet
Références : `SOUPs1` (0,3942 : 4,869 / 4,553) puis `PG_RDq_70_l915c_s1` de t3-redistill (**0,4107** : 5,087 / 4,616). Témoin `TM_ctl0` : la recette de `SOUPs1` relancée donne 4,869 / 4,553, identique.
| Fichier | Ce qui change | m / s exactes | m / s copies légères | sha256 | Score réel |
|---|---|---|---|---|---|
| `TM_wT2_4_a114` | perte de forme à T=2 × 4, déplacement × 1,14 | 4,860 / 4,363 | 4,302 / 4,032 | 492f49ac… | 0,3611 (perd) |
| `TM_t10h2` | bascules graduées (70 % à loss forte + 10 % à loss 1-3), ancienne cible | 4,859 / 4,501 | 4,303 / 4,146 | 48dd261d… | 0,3884 (neutre) |
| `TM_H4a_a090` | cible consensus de 4 sources, déplacement × 0,90 | 5,092 / 4,913 | 4,445 / 4,478 | 60cfff68… | 0,3953 (neutre) |
| `TM_H4a_w14` | cible consensus, terme de loss × 1,4, sans mise à l'échelle | 4,962 / 4,778 | 4,353 / 4,330 | 61ee1bd8… | en réserve |
| `TM_b6` | 6 blocs entraînés, recalé | 5,207 / 4,378 | 4,522 / 4,042 | b8aab9bc… | non envoyé |
| `TQ_t10a_s1` / `_s0` | bascules graduées sur la cible gagnante, 2 graines | 5,148 / 4,643 ; 5,150 / 4,665 | 4,514 / 4,264 ; 4,513 / 4,276 | 50944589… ; f1f6787d… | livrés |
| `TQ_Hqg_s1` / `_s0` | cible consensus sur la cible gagnante, 2 graines | 5,254 / 4,600 ; 5,214 / 4,577 | 4,623 / 4,223 ; 4,586 / 4,202 | 63bb44c4… ; 38d40a36… | livrés |
Cartographie du sommet, sans entraînement (déplacement de poids du gagnant multiplié par un facteur) :
| Fichier | Facteur | m / s exactes | m / s copies légères | sha256 |
|---|---|---|---|---|
| `PG_RDq_70_l915c_s1_a095` | × 0,95 | 4,692 / 4,375 | 4,007 / 4,009 | 85788754… |
| `PG_RDq_70_l915c_s1` | × 1,00 | 5,087 / 4,616 | 4,384 / 4,233 | (0,4107) |
| `PG_RDq_70_l915c_s1_a103` | × 1,03 | 5,303 / 4,747 | 4,596 / 4,355 | 89419671… |
| `PG_RDq_70_l915c_s1_a106` | × 1,06 | 5,503 / 4,867 | 4,795 / 4,468 | 3f6bde51… |
| `PG_RDq_70_l915c_s1_a110` | × 1,10 | 5,747 / 5,012 | 5,043 / 4,606 | dd3e46ef… |
| `SOUPs1_a105` (ancienne cible recalée) | × 1,05 | 5,202 / 4,751 | 4,571 / 4,359 | 38cd0cde… |
Leçons : un régulateur de loss en cours d'entraînement déforme la recette (témoin régulé à 5,44 / 5,40 : abandonné) ; aucune variante d'entraînement n'a battu la recette simple ; le gain vient de la source des images et des classes, et du calage de la loss.

## 0. Recette « RDL » : même oubli, dérive réduite (cible de t3-redistill à 70 %)
Cible `RDt24_70` : mes images (70 % basculées), classe fausse prédite par les modèles de t3-redistill. Entraînement plus long, ancre « mélanges » plus forte, loss moyenne maintenue par un régulateur. Référence : `PG_RDm3t24_70_l1016c` (0,3676 au serveur), 1 500 pas, ancre 3.
| Fichier | Pas / ancre | m / s exactes | m / s copies légères | Bascules | Mélanges : accord / KL | sha256 | Score réel |
|---|---|---|---|---|---|---|---|
| référence `PG_RDm3t24_70_l1016c` | 1 500 / 3 | 4,663 / 4,542 | 4,086 / 4,166 | 65,4 % | 0,726 / 0,294 | — | 0,3676 |
| `RDL_s3k_l6.pt` | 3 000 / 6 | 4,742 / 4,446 | 4,094 / 4,090 | 66,4 % | 0,783 / 0,166 | 58720244… | **0,3578** : sous son équivalent `PG_RDt24_70_l1016c` (0,3653) |
| `RDL_s3k_l12_v3.pt` | 3 000 / 12, poids 3 sur les vues | 4,602 / 4,914 | 4,523 / 4,924 | 60,1 % | 0,806 / 0,146 | 600274b2… | livré |
| `RDL_s5k_l12.pt` | 5 000 / 12 | 4,073 / 4,598 | 3,595 / 4,279 | 62,5 % | 0,838 / 0,096 | 5c9ddad4… | livré |
| `RDL_s5k_l6.pt` | 5 000 / 6 | 4,513 / 3,251 | 3,983 / 3,122 | 69,6 % | 0,813 / 0,131 | 2fc9e60c… | sonde (écart-type bas) |
**Verdict du serveur : la recette RDL ne paie pas** (0,3578 contre 0,3653), malgré une dérive presque divisée par deux. Diagnostic (`diag_shape.py`, sortie `h200:~/eac/t3/t3-methodes/diag/forme_RDL.md`) : sur les images basculées les deux fichiers sont identiques à 2 % près ; la différence est sur les images non basculées, que RDL laisse plus sûres d'elles (bonne classe à T=2 : 0,900 contre 0,867) et plus proches d'orig. Les autres fichiers RDL ne sont donc pas à envoyer.

Limite mesurée : à 3 000 pas, une ancre à 12 sans poids sur les vues ne laisse monter la loss qu'à 1,73, et une ancre à 25 à 0,5. Une ancre plus forte demande plus de pas.
Scores réels de repère (fichiers de l'orchestrateur et de t3-redistill sur ma recette) : 60 % 0,3009 ; 70 % 0,3159 ; 80 % 0,3179 ; classe de t3-redistill à 60 % 0,3273, à 70 % 0,3676, à 80 % **0,3713**.

## 1 bis. Série à loss égale à 60 % (un seul changement par fichier, loss moyenne régulée sur 4,22)
Référence : `PG_T85_f60_l814c` (0,3009) : 4,221 / 4,431 sur images exactes, 3,649 / 4,021 sur copies légères, max-prob T=2 0,674. Les variantes se comparent entre elles et au témoin `PGm_f60b`, dont l'écart-type est plus bas que la référence (3,99).
| Fichier | Changement | m / s exactes | m / s copies légères | Max-prob T=2 | KL mélanges | sha256 |
|---|---|---|---|---|---|---|
| `PGm_f60b.pt` | témoin : recette inchangée + régulateur | 4,275 / 3,990 | 3,699 / 3,664 | 0,653 | 0,242 | 3e798984… |
| `PGw15_f60b.pt` | masse fausse des images basculées plus étalée (classe dominante 0,15 à T=2 au lieu de 0,32) | 4,309 / 4,233 | 3,822 / 3,899 | 0,535 | 0,243 | c9730ec8… |
| `PGw2_f60b.pt` | encore plus (0,09) | 4,353 / 4,240 | 3,889 / 3,915 | 0,456 | 0,245 | 1ce4cfe5… |
| `PGs70m_f60b.pt` | images non basculées plus étalées (bonne classe à T=2 : 0,64 au lieu de 0,83) | 4,281 / 4,038 | 3,728 / 3,689 | 0,613 | 0,249 | abab1deb… |
| `PGs48m_f60b.pt` | encore plus (0,44) | 4,292 / 4,094 | 3,782 / 3,718 | 0,543 | 0,256 | 52608721… |
| `PGv2_f60b.pt` | poids 2 sur les vues augmentées | 4,263 / 3,856 | 3,923 / 3,707 | 0,636 | 0,280 | f4f35759… |
| `PGv3_f60b.pt` | poids 3 | 4,263 / 3,825 | 4,029 / 3,740 | 0,625 | 0,317 | e04e22fa… |
Sans régulateur, livrés avant : `PGs70_f60_l814c.pt` (4,034 / 4,290, copies 3,522 / 3,886, sha 14800c88…, lisible contre 0,3009), `PGs48_f60_l814c.pt` (3,793 / 4,076, sha 784387a5…), `PGo_f60_l814c.pt` (classe fausse = second choix d'orig : 4,765 / 4,770, sha 4a331523…), `PGrd1_f60_l814c.pt` (images et classe de t3-redistill : 4,973 / 4,847, sha 428de691…).

## 2. Tableau de sécurité : la même recette entraînée sans la réserve
| Recette | m / s / bascules (2 802 images) | Bascules dans la réserve | KL réserve | Rétention réserve | Flou fort (orig 0,867) | Verdict |
|---|---|---|---|---|---|---|
| Cibles du 1er agent, sans ancre | 0,409 / 0,738 / 12,1 % | 0,4 % | 0,931 | 0,516 | 0,837 | non |
| Ancres croisées (somme de deux déplacements) | 0,369 / 0,564 / 5,4 % | 0,4 % | 1,037 | 0,489 | 0,855 | non |
| Pseudo-gold 15 %, bascules peu confiantes (`PG_T85`) | 0,327 / 0,840 / 13,7 % | 0,0 % | 0,107 | 0,904 | 0,817 | sûr |
| 10 % de bascules confiantes | 0,591 / 1,789 / 10,2 % | 0,6 % | 0,101 | 0,903 | 0,822 | sûr |
| 20 % | 1,144 / 2,340 / 20,3 % | 1,9 % | 0,152 | 0,852 | 0,779 | sûr |
| 30 % | 1,591 / 2,545 / 29,7 % | 3,6 % | 0,368 | 0,705 | 0,662 | limite |
| 50 % | 2,338 / 2,593 / 48,1 % | 15,3 % | 0,886 | 0,449 | 0,423 | non |
| 20 % à loss 8-14 (`PG_T85_f20_l814c`) | 2,045 / 4,185 / 20,4 % | 2,4 % | 0,148 | 0,850 | 0,746 | sûr |
| 30 % + ancre sur vues très dégradées (`PD_f30`) | 1,406 / 2,330 / 29,6 % | 0,4 % | 0,076 | **0,926** | 0,855 | sûr |
| Recette de `PG_T70` (référence : a passé la porte, 0,0323) | 0,487 / 0,815 / 21,9 % | 0,7 % | 0,320 | 0,752 | 0,722 | référence |
| 30 % à loss 8-14 (`PG_T85_f30_l814c`) | 2,976 / 4,689 / 30,7 % | 3,7 % | 0,211 | 0,795 | 0,695 | envoyable |
| 40 % à loss 8-14 (`PG_T85_f40_l814c`) | 3,747 / 4,891 / 40,4 % | 6,7 % | 0,303 | 0,716 | 0,625 | limite |
| 30 % × 8-14, images non basculées intactes (`PGh`) | 3,101 / 4,838 / 30,7 % | 2,3 % | 0,124 | **0,869** | 0,722 | meilleur compromis mesuré |
| 30 % × 8-14, cumul `PGho` | 3,178 / 4,956 / 30,6 % | 2,4 % | 0,092 | **0,894** | 0,779 | meilleur |
| `PGho` 50 % | 4,901 / 5,278 / 50,2 % | 5,7 % | 0,244 | 0,758 | 0,670 | envoyable |
| `PGho` 60 % | 5,649 / 5,185 / 59,6 % | 9,3 % | 0,353 | 0,670 | 0,607 | limite |
| `PGho` 70 % | 6,345 / 4,923 / 69,2 % | 13,9 % | 0,517 | 0,568 | 0,524 | marge de porte faible |
| `PGh` 50 % | 4,600 / 5,040 / 50,0 % | 8,0 % | 0,348 | 0,682 | 0,586 | inférieur au cumul |
| `PGh` 60 % | 5,209 / 4,926 / 59,2 % | 14,0 % | 0,545 | 0,556 | 0,476 | inférieur au cumul |
| 30 % × 8-14, images et classe de t3-redistill (`PGrd1`) | 3,085 / 4,881 / 30,1 % | 3,2 % | 0,152 | 0,841 | 0,764 | retient mieux que mon sélecteur (0,795) |
| 30 % × 8-14, ancre sur images générées (`PGx`) | 2,903 / 4,606 / 30,6 % | 2,1 % | 0,126 | 0,869 | 0,742 | égal à PGh, avec données externes |
| 30 % × 8-14, classe fausse = second choix d'orig (`PGo`) | 3,095 / 4,855 / 30,7 % | 2,7 % | 0,159 | 0,839 | 0,738 | gain de rétention |
| 30 % × 8-14, quota par classe (`PGq`) | 2,931 / 4,650 / 30,6 % | 3,3 % | 0,200 | 0,806 | 0,693 | neutre |
| 30 % à loss 12-18 (croisement `PG_T85_f30_l1218c`) | 3,885 / 6,152 / 30,7 % | 5,2 % | 0,225 | 0,774 | 0,681 | envoyable |
| 30 % à loss 8-14 + rotations dans les vues (`PGr`) | 2,758 / 4,439 / 30,6 % | 3,2 % | 0,190 | 0,814 | 0,721 | pas de gain net |
| 30 % à loss 8-12 + ancre dégradée, sélecteur instabilité (`PS30_err`) | 2,081 / 3,530 / 29,8 % | 0,4 % | 0,069 | 0,932 | 0,850 | sûr, mais oublie moins sur les copies |
| 50 % + ancre sur vues très dégradées | 1,453 / 1,855 / 41,1 % | 5,7 % | 0,563 | 0,603 | **0,854** | porte protégée, rétention à améliorer |
| Toutes les images incertaines, sans bascule | 0,857 / 0,432 / 2,0 % | 1,0 % | 1,452 | 0,404 | 0,650 | non |

Règle de porte du vérificateur (depuis la sonde `G10_PG_T70` = 0,0208, qui montre environ 10 points de marge) : « go » si la précision de réserve (100 % moins la colonne « bascules dans la réserve ») reste au-dessus de 0,91 environ. Toutes les recettes jusqu'à 30 % sans ancre et 50 % avec ancre passent ; ce qui limite désormais est la rétention.

**Leçon du score 0,0916** : l'ancre sur vues très dégradées protège très bien la réserve, mais elle affaiblit la loss sur les copies géométriques (rotation : 0,76 / 1,98 contre 1,37 / 3,06 sans elle), et les copies comptent pour la moitié de la note d'oubli. La recette gagnante garde donc l'ancre « mélanges » seule.

Lecture : une bascule reste sur l'image visée ; l'incertitude se propage à toute la classe. Au-delà de 20 à 30 % de bascules, les classes elles-mêmes cèdent, sauf si l'on ancre le modèle sur des versions très dégradées des images (là où orig répond déjà comme devant une image jamais vue).

**Limite de la réserve, constatée au serveur** : `PGho` à 60 % retient mieux que la recette simple sur la réserve et sur les images générées, mais fait 0,2701 contre 0,3009. La réserve mesure la rétention et la marge de porte, pas le score de prédiction. Une recette qui gagne sur la réserve n'est donc qu'une candidate, pas une amélioration acquise.

## 2 bis. Rétention sur images générées jamais vues (outil `retproxy.py` de t3-nonmembres)
Même formule que le serveur, sur cinq jeux d'images générées à partir des images à oublier ou de leur légende. Sert de second témoin, indépendant de ma réserve.
| Fichier | Image-vers-image 0,70 | Image-vers-image 0,85 | IP-Adapter | Texte-vers-image SDXL | Texte-vers-image FLUX |
|---|---|---|---|---|---|
| `PG_T85_f30_l814c` (0,1962) | 0,472 | 0,576 | 0,554 | 0,825 | 0,732 |
| `PGho_f30_l814c` | 0,523 | 0,633 | 0,628 | 0,875 | 0,804 |
| `PG_T85_f50_l814c` (0,2779) | 0,302 | 0,409 | 0,391 | 0,763 | 0,580 |
| `PGho_f50_l814c` | 0,319 | 0,444 | 0,418 | 0,771 | 0,638 |
| `PGh_f50_l814c` | 0,314 | 0,425 | 0,413 | 0,799 | 0,621 |
| `PGho_f60_l814c` | 0,240 | 0,358 | 0,346 | 0,739 | 0,578 |

## 3. Simulateur hors ligne (`offsim.py`, sans entraînement)
Une famille de dégradations laissée de côté joue le gold, la cible est construite avec les quatre autres, et on applique la formule officielle à la cible. Ce n'est pas le vrai gold. Il a servi à choisir « peu de bascules, très confiantes » ; les scores réels disent depuis que le vrai gold a une loss bien plus haute que ce faux gold (0,63 / 1,76), donc ses optimums (5 à 10 % de bascules) sont trop bas.
