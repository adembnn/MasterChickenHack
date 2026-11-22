# T1 · guide oral : comprendre, présenter, répondre

État au 22/11, 11 h. Décision débloquée : ce qu'Adem dit du track 1 devant le jury, et comment il répond.
Diapos concernées dans `Présentation_AI.pptx` : 3 (Public score 0.823), 4 (pipeline), 5 (progression et plafond).
Sources : issue #3, `reports/t1-tableau.md`, `reports/t1-t1-reel.md`, `reports/t1-t1-assemblage.md`, `reports/t1-t1-residus.md`, leaderboard relu à 10 h 55.

## Partie 1 · Comprendre, depuis zéro

**Image.** Une image de la tâche fait 512 × 512 pixels. Un pixel = trois nombres de 0 à 255 (rouge, vert, bleu).

**Bit et message.** Un bit vaut 0 ou 1. Le message caché est une suite de bits, par exemple 100 bits. On connaît le message de chaque image : il est donné.

**Filigrane.** L'outil de filigrane (l'« embedder ») modifie très légèrement les pixels, de quelques unités sur 255, pour y écrire le message. L'œil ne voit rien. Trois outils : PixelSeal, TrustMark, StegaStamp.

**Décodeur.** Réseau fourni avec chaque outil. On lui donne une image, il rend pour chaque bit un score : positif = « je lis 1 », négatif = « je lis 0 ». Sur une image propre, il tombe juste une fois sur deux (50 %, le hasard). Il est fait pour lire *un* message sur *toute* l'image, pas pour dire *où* le message se trouve. C'est tout le défi.

**Masque.** On rend une grille de 128 × 128 cases. Une case couvre 4 × 4 pixels. Chaque case vaut 1 (filigrané) ou 0 (propre).

**IoU et mIoU, avec des chiffres.** Sur une image, 100 cases sont vraiment filigranées. On en prédit 120, dont 90 justes.
- IoU de la classe filigranée = cases communes / cases couvertes par l'un ou l'autre = 90 / (100 + 120 − 90) = 90 / 130 = **0,69**.
- IoU de la classe propre : presque toutes les autres cases sont justes, environ **0,998**.
- mIoU de l'image = moyenne des deux = **0,84**. Puis on fait la moyenne sur les 1 020 images.
Ce qu'il faut en retenir : la petite zone est celle qui coûte. Rater une petite zone fait tomber l'image à environ 0,5.

**Les deux familles d'images.**
- **Pleine** : l'image est filigranée partout, puis on a effacé ou collé un objet. Il reste un **trou** propre. Il faut dessiner le trou.
- **Partielle** : l'image est propre, sauf un **rectangle** (outpainting : l'image filigranée a été agrandie, le filigrane n'est que dans l'ancienne image) ou un **objet** filigrané collé.
Environ une image sur deux est pleine. C'est pour ça que même le rectangle parfait ne fait que 0,66 : sur une pleine, le rectangle, c'est l'image entière, et le trou est raté.

**Segmenteur, U-Net.** Un réseau de neurones qui prend l'image et rend un score par case (positif = filigrané). Le U-Net a deux moitiés : un encodeur qui résume l'image en caractéristiques de plus en plus grossières, un décodeur qui remonte à la pleine taille pour donner une carte. Notre encodeur est un ResNet-34, un réseau public déjà entraîné sur des millions de photos.

**Entraîner, affiner, plis.**
- On n'a que 170 vraies images annotées par filigrane. C'est peu. Alors on **fabrique** des exemples : on prend des images, on les filigrane nous-mêmes avec l'outil officiel et le bon message, puis on colle des morceaux propres aux formes des vrais trous. On connaît donc la vérité exacte.
- Le réseau apprend 8 000 pas sur ces exemples fabriqués, puis on l'**affine** (courte fin d'entraînement) sur les vraies images.
- **5 plis** : on coupe les vraies images en 5 groupes. On entraîne 5 modèles, chacun sans un groupe. Chaque image est ainsi prédite par un modèle qui ne l'a jamais vue. Au test, on moyenne les 5 modèles.

**SAM (Segment Anything, Meta).** Réseau public qui découpe une photo en objets (une personne, une chaise, un pan de ciel), sans rien savoir des filigranes.

**Score z.** On découpe un morceau de l'image, on le donne au décodeur, on compare ce qu'il lit au vrai message. z dit à combien d'écarts-types on est au-dessus du hasard. Hasard : z ≈ 0 (± 1). Morceau bien filigrané : z ≈ 10 (le maximum pour 100 bits). On garde un rectangle si z > 6,5.

**Intervalle de confiance à 95 %.** On compare deux fichiers sur les mêmes images, image par image. On tire 10 000 fois des images au hasard parmi elles et on recalcule l'écart. Si 95 % des tirages donnent un écart du même signe (l'intervalle n'inclut pas 0), le gain est réel et pas dû à la chance.

## Partie 2 · Le fil, en trois idées
Si tu ne retiens que ça, tu tiens tout l'oral :
1. **Le signal existe mais il est flou.** Dans la zone filigranée, le décodeur lit 85 % des bits justes ; dehors, 50 %. Mais sa carte native fait 8 × 8 cases : beaucoup trop grossière.
2. **On apprend la localisation, puis on corrige les formes.** Un U-Net entraîné sur des exemples fabriqués *fidèlement* (deux découvertes : la toile et la pleine résolution), puis SAM et le décodeur pour donner des bords nets.
3. **On mesure avant d'envoyer, et ça a tenu.** Chaque envoi avait une prévision écrite. Le plafond restant est une limite du signal : les tout petits trous.

## Partie 3 · Ce que tu dis, diapo par diapo
Temps visé : 2 minutes pour le track 1 (environ 40 s par diapo). Si le temps manque, dis seulement les phrases en **gras**.

### Diapo 3 · « Public score: 0.823 · 1st on Task 1 » (40 s)
Montre la boîte HYPOTHESIS, puis la colonne TEST, puis le grand 0.823.
> **"Task 1: each test image carries an invisible watermark, partly erased or partly added by an edit. We return a 128 by 128 mask saying where the watermark still is."**
> "These watermarks are built to read one message from the whole image, not to say where it is. But we measured that the decoder reads 85% of the bits correctly inside the watermarked zone, and 50%, pure chance, outside. So the signal is there."
> **"We turned that signal into a pixel map: a segmentation network, then shape correction, and every submission tested before sending. We went from 0.635 to 0.823, first on this task."**
> (en montrant la ligne du bas) "One thing to know: a perfect bounding box only gets 0.66. The score is decided by the shape of the holes."

### Diapo 4 · « Route, segment, snap to shapes, threshold » (45 s)
Suis les cartes de gauche à droite avec la main.
> **"Four stages."**
> "One: routing. The decoder's bit accuracy on the whole image tells us if the image is fully watermarked with a hole, or clean with a watermarked patch. It's right on 100% of our held-out PixelSeal and TrustMark images."
> **"Two: a U-Net, trained on edits we fabricate ourselves with the official watermarking tools, then fine-tuned on the real images, five folds, several models averaged."**
> **"Three: snap to real shapes. For objects, SAM cuts the image into objects and each object takes the majority vote of our map. For rectangles, we search thirteen thousand crops for the one where the decoder reads the message best, then align it to the pixel."**
> "Four: thresholds, tuned only on images the model never saw."
> (en bas) "Result per watermark: StegaStamp 0.93, TrustMark 0.81, PixelSeal 0.71. Three watermarks, three behaviours."

### Diapo 5 · « Two discoveries, then a signal limit » (40 s)
Montre le graphique de gauche à droite, puis la boîte claire, puis la boîte bleue.
> **"Two discoveries moved the score."**
> "First, the canvas: the organisers watermarked a canvas about 32 pixels larger than the image, then cropped it. Once our fabricated examples did the same, StegaStamp went from 0.66 to 0.82 without seeing a single real image."
> "Second, the network's first layer reads every pixel instead of one in two. That's when PixelSeal finally started to learn."
> **"Where we cap: 57% of the remaining error is the hole in fully watermarked PixelSeal and TrustMark images. Under 32 pixels, the decoder itself reads chance. That's a limit of the signal, not of our post-processing."**
> Transition : "And that's the method we used on every task: measure first, then decide." (puis passe au track 2.)

### Ce qu'il ne faut pas dire
- « On n'a jamais utilisé les vraies images. » Faux : on affine sur les vraies images.
- « On a mesuré le temps de calcul. » Jamais mesuré.
- « On est clairement devant Incoming PhD. » Pas établi : 0,82 contre 0,82, bruit du classement ± 0,01.
- « On a inventé un nouveau modèle. » Non : U-Net et SAM sont publics. Ce qui est à nous : la fabrication fidèle, l'assemblage, la discipline de mesure.

## Partie 4 · Répondre aux questions
Méthode pour chaque réponse : **une phrase de réponse, un chiffre, stop.** S'ils veulent plus, ils relancent. Une réponse courte avec un chiffre montre que tu as compris ; une longue réponse floue montre l'inverse.

Si tu ne sais pas : ne devine pas. Dis ce qu'on a mesuré à côté, puis passe la main.
> "I don't have that number in mind. What we measured is [chiffre proche]. Titouan or Yassir can complete." / "We didn't measure that, so I won't guess."
Un jury respecte plus « on ne l'a pas mesuré » qu'une invention qu'il démonte en une question.

**Q1. Why not Watermark Anything, which is built for localization?**
> "It localizes its own watermark. Here the watermarks are fixed: PixelSeal, TrustMark, StegaStamp. So we used their own decoders as the signal, and learned the localization on top."

**Q2. How do you know you didn't overfit the public leaderboard?**
> "Nothing was tuned on the public score. We kept 102 training images aside, used five-fold cross-validation, and wrote a prediction before each submission. From our fifth to our 35th candidate, we predicted plus 0.065 and got exactly plus 0.065."

**Q3. Why is PixelSeal so much harder?**
> "PixelSeal was trained against a network that tries to detect it, so its trace is weak and spread out. StegaStamp leaves a strong pattern: 0.93 against 0.71."

**Q4. What was the key idea?**
> "Making our fabricated training data faithful. The real watermark was computed on a larger canvas. Reproducing that took StegaStamp from 0.66 to 0.82 with zero real images."

**Q5. What is the routing exactly?**
> "We run the decoder on the whole image. A fully watermarked image gives most bits right, a partial one far fewer. One threshold, around 75% correct bits, splits them: no error on our held-out PixelSeal and TrustMark images, 94% on StegaStamp."

**Q6. How do you find the rectangles?**
> "We try about thirteen thousand crops and keep the one where the decoder's reading is furthest above chance, a z-score above 6.5. Then we adjust each edge to the pixel by re-watermarking: inside the true frame, the tool adds less, because the message is already there. On TrustMark rectangles, that went from 0.89 to 0.94."

**Q7. Did you use external data?**
> "Only public pretrained weights: ResNet and SAM. Our training examples are built from the hackathon images and the official watermarking tools. COCO was only used in the very first version."

**Q8. What would give you plus 0.03?**
> "Drawing the holes of fully watermarked PixelSeal and TrustMark images as well as StegaStamp: from 0.73 and 0.76 up to 0.85 would be worth plus 0.035. We tried hole-specialist models; their advantage vanished after fine-tuning."

**Q9. What did you try that failed?**
> "Training-free maps from the decoder, at 0.5 at best. Morphology and finer thresholds: neutral. Adapting the model on each test image: minus 0.02, it just inflates the zone. Every idea had a number before we kept or dropped it."

**Q10. Is it usable in the real world? How fast is it?**
> "We didn't measure speed, it's a competition pipeline with several networks. What transfers directly: with the decoder's z-score, you can prove whether a given region carries the owner's message. That's what you need to contest a false attribution."

**Q11. Why does it matter?**
> "If you can't localize the watermark, anything added to a watermarked image gets attributed to the watermark owner. Someone can frame them with content they never made."

**Q12. How did you organise the work?**
> "Several AI agent sessions in parallel on GPUs, each on one part. One rule: whoever builds a file never validates it. A separate checker recomputes everything, and one single session submits."

## Partie 5 · Les chiffres à avoir en tête
| Chiffre | Ce qu'il veut dire |
|---|---|
| 0,823 | notre meilleur score public, 1er du track 1 |
| 0,635 | notre premier envoi, samedi 17 h 25 |
| 85 % / 50 % | bits justes du décodeur dans la zone / dehors |
| 0,25 / 0,66 | masque constant / rectangle parfait |
| 0,93 · 0,81 · 0,71 | StegaStamp · TrustMark · PixelSeal |
| 0,66 → 0,82 | StegaStamp avant / après la correction de toile, sans vraie image |
| 57 % | part de l'erreur restante due aux trous des pleines PixelSeal et TrustMark |
| 32 px | taille sous laquelle le décodeur ne lit plus que le hasard |
| +0,065 / +0,065 | gain prévu / gain obtenu, du candidat 5 au 35 |
| 102 | images de réserve, jamais utilisées pour apprendre ni régler |
| 3e, 0,96 | classement général (Murmell 0,99, teambuilders 0,97) |
