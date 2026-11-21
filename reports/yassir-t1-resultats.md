# T1 · Premiers résultats réels (agent de Yassir)

21 novembre 2026, 16h30 UTC, sur la H200 Nebius. 510 images de train, score officiel mIoU
(`wmloc.utils.score`), seuil balayé de 0,30 à 0,74 par pas de 0,02 pour chaque méthode.
Ce ne sont plus des propositions, ce sont des mesures.

## Le classement des variantes

- Tout à zéro (référence du kit) : **0,2518**
- Décodage, canevas, fenêtre 256, pas 64 : **0,4627**
- Décodage, canevas, fenêtre 192, pas 48 : 0,4623
- Décodage, canevas, fenêtre 320, pas 80 : 0,4583
- Décodage, canevas, carte par position à poids 1,0 : 0,4643
- Décodage, canevas, sans carte par position : 0,4147
- Décodage, crop, fenêtre 256, pas 64 : 0,3943

Deux conclusions tiennent.

**Un, l'hypothèse d'Adem est confirmée.** Le mode crop, qui agrandit la fenêtre et casse l'échelle
du motif, perd 6,8 points sur le score global. Le détail est plus parlant que le total : StegaStamp
tombe de 0,4484 à 0,2589, soit 19 points, parce que son décodeur travaille à 400 × 400 et qu'un
agrandissement change complètement la période du motif. PixelSeal et TrustMark résistent mieux,
ils perdent moins de 3 points. Le canevas, qui replace la fenêtre à sa position exacte, corrige ça
sans rien entraîner.

**Deux, la carte par position vaut 4,8 points.** Sans elle, PixelSeal tombe à 0,3355, alors qu'à
poids 1,0 il remonte à 0,4719. C'est cohérent avec le fait qu'elle est calculée sur l'image entière,
donc à la bonne échelle par construction. Le poids 0,5 et le poids 1,0 donnent la même chose à
0,001 près, donc on garde 0,5 par prudence.

## Par méthode, en canevas 256/64

- PixelSeal : 0,4704, seuil 0,56
- TrustMark : 0,4694, seuil 0,58
- StegaStamp : 0,4484, seuil 0,58

Les trois sont proches, ce qui est rassurant : aucune méthode n'est cassée, et le seuil optimal est
dans une bande étroite, 0,56 à 0,58, donc le signal est bien calibré. Attention, un seuil aussi
haut veut dire que la carte est franche : peu de cellules hésitent entre propre et filigrané.

## Soumission prête

`submission_t1.pt` est écrite sur la VM : 1020 masques en 128 × 128 booléen, 16,7 Mo, tous les noms
du test présents une fois, format conforme. Part de cellules à 1 : 0,497 PixelSeal, 0,360 TrustMark,
0,390 StegaStamp.

Elle n'est pas envoyée : il manque la clé API. Dès qu'elle est là, `submission_template.py` avec
`SUBMIT = True` fait l'envoi, ou un simple appel `submit(path)`.

## Ce que je ferais ensuite, dans l'ordre

1. Segmentateur supervisé sur les 510 masques connus, en prenant l'image et la carte de décodage en
   entrée. C'est là que se trouve le gain suivant, pas dans le réglage des fenêtres, qui plafonne
   autour de 0,46.
2. Séparer par type de retouche, puisque le train étiquette `full_watermark`. Une image dont le
   filigrane couvre tout puis a été restaurée localement n'a pas la même erreur qu'une greffe
   d'objet.
3. Vérifier le score de test à la première soumission, pour savoir si le train et le test se
   ressemblent. Si l'écart est gros, c'est le segmentateur qui généralise mal, pas le décodeur.

## Reproductibilité

Tout tourne sur la VM `green-starfish-instance-6`, H200 143 Go, 16 vCPU. Environnement :
`uv venv` python 3.12, torch 2.5.1 cu124, kit officiel, `download_models.py` exécuté une fois.
Le train complet passe en 90 secondes en canevas et 50 secondes en crop, donc les balayages sont
bon marché et il faut en faire plus, pas moins.
