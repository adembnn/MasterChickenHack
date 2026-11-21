# Deux essais de plus, deux résultats négatifs propres (agent de Yassir)

21 novembre, 19h30 UTC. Les deux tournaient en parallèle sur les deux H200.

## T1, segmentateur sur données synthétiques, test de la correction bf16

J'ai voulu vérifier l'hypothèse du critique d'équipe : le stem en bf16 détruit le résidu fin de
PixelSeal, et il faudrait soit un stem en fp32, soit une entrée passe-haut amplifiée.

Montage : bases propres prises dans les zones non filigranées des images de train (36 bases
exploitables, le reste n'a pas de zone assez propre), formes tirées des vrais masques, vrai embedder
PixelSeal, 1200 échantillons, U-Net 256 vers 128, puis évaluation sur la réserve réelle (34 images,
formes jamais vues).

- Bras bf16, entrée image seule : **0,2275** de mIoU sur la réserve.
- Bras fp32, entrée image plus passe-haut : **0,2275** exactement.

Deux chiffres identiques au centième et une perte bloquée à 0,95, soit la valeur d'un réseau qui
prédit une constante. Autrement dit, dans mon montage, **aucun des deux bras n'apprend quoi que ce
soit**. La comparaison ne prouve donc rien sur bf16 : elle dit que mon générateur ou mon architecture
ne laisse pas passer le signal, pas que le correctif est inutile.

Ce n'est pas la peine d'insister sur cette voie : la version de l'équipe (`seg2.py`, tronc ResNet-34,
6000 échantillons, tirage des formes et des échelles dans les lois réelles) est plus avancée et c'est
elle qui porte le 0,697. Mon montage n'apporte rien de plus.

## T3, quatrième essai : étaler la sortie sur les images mémorisées

Idée inverse des deux premiers essais. Au lieu de pousser vers une distribution structurée, on étale :
cible = 90 % uniforme mélangée à 10 % de la sortie d'origine, sur les deux derniers blocs seulement,
4 époques, pénalité L2.

Mesures sur les 3500 images à oublier :

- confiance moyenne : 0,9997 vers **0,019**
- entropie : 0,097 vers **4,58** (maximum possible 4,61)
- écart au modèle original : **4,24** en KL
- accord de classe avec l'original : 13,5 %

**Pas soumis**, et je l'assume. Un modèle qui répond presque uniforme sur ces images s'éloigne
probablement trop de ce que donnerait un modèle réentraîné, qui reste un classifieur précis même sur
une image jamais vue. Et je n'ai aucun moyen de mesurer la porte d'utilité ici : mon proxy (vues
augmentées des images à oublier) est invalide, puisqu'une vue augmentée reste la même image à oublier.

## Ce qu'il faudrait pour trancher T3

Un jeu de contrôle, c'est-à-dire des images que le modèle original n'a jamais vues et dont on connaît
l'étiquette. Sans lui, on avance à l'aveugle et chaque soumission coûte une fenêtre de 30 minutes.
C'est le seul achat qui changerait la donne sur cette tâche.
