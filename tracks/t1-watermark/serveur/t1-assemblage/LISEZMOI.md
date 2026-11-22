# t1-assemblage : quels fichiers sont des candidats

Seuls les fichiers `submission_candidat_<n>.pt` écrits par `compose.py` sont des candidats, et seulement ceux qui ont un feu vert écrit du vérificateur sur l'issue #3 sont à envoyer.
Les fichiers `assemblage_<n>.pt` sont des sorties intermédiaires de `assemble.py` (sources restreintes pour aller vite) : ne pas les envoyer.

Attention aux anciens numéros : jusqu'au 22/11 à 0 h 30, `assemble.py` écrivait aussi ses sorties intermédiaires sous le nom `submission_candidat_<n>.pt`.
Intermédiaires sous l'ancien nom : 10, 12, 14, 16, 18, 21, 23, 25, 27, 29. Candidats composés : 11, 13, 15, 17, 19, 20, 22, 24, 26, 28, 30, 31 (31 = 30).
Le 27 a été envoyé par erreur de lecture le 22/11 à 0 h 50 (public 0,7935) : c'est un intermédiaire dont le PixelSeal est plus faible (réserve 0,626 contre 0,647 dans le 30).

Meilleur candidat : voir le dernier commentaire de la session sur l'issue #3 et `reports/t1-tableau.md`.
