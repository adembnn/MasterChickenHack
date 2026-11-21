# T1 : localisation de filigrane

- **Responsable :** à attribuer
- **Consigne officielle :** `reports/sources/Watermark_Localization.txt`
- **Recherche :** `reports/t1-watermark-etat-de-lart.md`
- **Issue GitHub :** #3 (y poster chaque lot)

## Ce qu'on nous demande
Pour 1020 images de test 512×512, on rend un masque 128×128 qui dit quelles zones portent encore le filigrane (PixelSeal, TrustMark ou StegaStamp) après une retouche : outpainting, retrait, ajout ou greffe d'objet. Le message caché et la méthode sont donnés pour chaque image. Le kit fournit les décodeurs. Score : moyenne par image de l'IoU de la zone propre et de l'IoU de la zone filigranée.

## État
| Date | Méthode | Score local | Score public | Qui |
|---|---|---|---|---|
| 21/11 17:25 | seg v2 : PixelSeal tronc décodeur + pyramide (réserve 0,57), TrustMark U-Net (0,56), StegaStamp U-Net (0,75), vraies images hors réserve mêlées aux lots, TTA symétrie, seuil 0 | réserve 0,627 | **0,635** (soumission 1103) | agent d'Adem |
| 21/11 18:18 | seg v3 (session t1-reel, run mixr) : StegaStamp 0,82 et TrustMark 0,60 = U-Net ResNet-34 entraîné sur exemples fabriqués à partir de vraies images du hackathon + vraies images, 5 plis moyennés ; PixelSeal inchangé (0,57) | réserve 0,663 | **0,674** (soumission 1190) | agent d'Adem + t1-reel |

## Prochain lot
À définir par le responsable.
