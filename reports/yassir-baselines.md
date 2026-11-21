# Baselines des trois tracks (agent de Yassir)

Trois décisions débloquées, trois scripts dans `tracks/*/`. Code écrit sans GPU, jamais exécuté sur
JURECA : à traiter comme des points de départ à vérifier, pas comme des résultats.

## T1 · Le décodeur est un localiseur

**Décision débloquée : on peut démarrer T1 aujourd'hui, sans entraînement ni GPU, parce que le
message est donné pour chaque image de test.**

Dans `wmloc/watermarks.py`, les trois extracteurs remontent l'image à une résolution de travail fixe
(256 PixelSeal, 245 TrustMark, 400 StegaStamp) et rendent des logits par bit. Le taux d'accord entre
les bits décodés et le message connu vaut 50 % sur une zone propre et monte sur une zone filigranée.
C'est donc directement la carte de localisation.

Deux signaux à sommer :
- fenêtre glissante 256 avec pas de 64, décodage global par fenêtre : précis, 25 passages par image ;
- `per_position_logits` pour PixelSeal et TrustMark : carte grossière (8×8) mais quasi gratuite.

`tracks/t1-watermark/decode_localize.py` balaie le seuil par méthode sur les 510 images d'entraînement
où le masque est connu, puis écrit `submission.pt` au format officiel. Commencer par
`--calibrate --limit 30` pour lire un mIoU de train en deux minutes.

Limite connue : les extracteurs moyennent les logits sur toute l'image, donc une fenêtre trop grande
dilue et une fenêtre trop petite n'est plus décodable après redimensionnement. C'est le seuil et la
taille de fenêtre que la calibration doit régler.

## T2 · Les références sont gratuites et étiquetées

**Décision débloquée : ne pas toucher aux 33 endpoints avant d'avoir réglé la décision sur les 9
références, qui ne coûtent aucune requête.**

`practice_answers.csv` donne la vérité des `ref-*` :
- même modèle : ref-01, ref-03 (system prompt), ref-05 (persona), ref-06 (watermark), ref-08 (température 1.0) ;
- autre modèle : ref-02 (fp8), ref-04 (LoRA 0.7), ref-07 (LoRA 1.5), ref-09 (LoRA 0.3).

36 paires étiquetées, 10 positives. On apprend donc les deux seuils (famille et modèle) au lieu de
les deviner, et on vérifie qu'on sépare bien quantification, LoRA, prompt système et watermark avant
de consommer un budget irréversible de 100 réponses par endpoint.

`tracks/t2-model-equality/solve_refs.py` : 100 prompts construits pour les six axes annoncés,
similarité par paire (correspondance exacte, ratio de séquence, écart de longueur), moyenne tronquée
au décile bas, balayage des seuils maximisant le MCC sur les références, puis classification des 33.

Rappel de pondération : 10 % famille, 90 % modèle. Tout l'effort doit aller au seuil fin à
l'intérieur d'une famille, pas au regroupement grossier.

## T3 · La cible n'est pas la destruction

**Décision débloquée : ne pas partir sur une montée de gradient nue, elle fait perdre sur les deux
termes du score.**

Le gold est la moyenne de modèles réentraînés sans les images à oublier : il garde une distribution
structurée, avec parfois la bonne classe, parfois une classe fausse précise. La composante de
prédiction est une KL par image normalisée par la KL entre l'original et le gold, donc un modèle qui
répond au hasard perd tout.

`tracks/t3-unlearning/unlearn_distill.py` distille la prédiction du modèle original sur une vue très
recadrée et floutée vers l'image d'origine. Cette prédiction n'a pas été mémorisée, elle ressemble
donc à ce que dirait un modèle qui n'a jamais vu l'image. Quatre modes, du plus prudent au plus
agressif, avec arrêt précoce et diagnostics avant/après (entropie moyenne, accord avec la vue
augmentée, part des prédictions qui changent).

## Blocage commun

Aucune soumission n'est possible sans la clé API du leaderboard : `POST /submit/{task_id}` exige
l'en-tête `X-API-Key`, et `/33-model-equality/v1/models` aussi. Recherche faite dans la boîte de
Yassir le 21/11 à 16h (SprintML, Dziedzic, CISPA, leaderboard, API key) : aucune trace. La clé doit
venir du formulaire d'équipe ou du stand, elle n'est pas récupérable depuis le serveur.
