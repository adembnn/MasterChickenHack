# T2 · Format du cycle 2 : 25 prompts distincts × 1 (RUT) contre 5 prompts × 5 tirages, sur jumeau Qwen2.5-1.5B

Zéro requête. `fmtgen.py` (h200, GPU) génère, sur 60 prompts ouverts (20 récits, 20 avis, 20 explications, « about 250 words »,
256 tokens, T = 0,75) : 16 réponses de référence de la base par prompt et 5 réponses par monde. Mondes : **copies** = base,
base + « You are a helpful assistant. » (sysH), base + prompt Acme (sysA), base à T = 1,0 ; **variantes** = fp8 W8A8 e4m3
per-tensor, LoRA r=8 q/v (entraînée sur des réponses de la base à un style imposé, autre corpus que le banc) à deux forces :
« loraEq » (J par token 0,034, soit celui de la fp8 0,031 : l'équivalent de LoRA 0,3 d'après le rapport v4) et « loraHalf »
(J = 0,010). `fmtan.py` : 1500 tirages par cellule, prompts tirés dans la moitié « évaluation » du banc, tirages des
réponses avec remise. Deux statistiques RUT-like contre la référence locale de la base sans prompt système : écart moyen
normalisé de log p_base par token (Stouffer) et Cramér-von Mises sur les rangs. Seuil = quantile 99 % du monde « base ».
## Résultats (`fmtan.out`)
Décalage moyen de log p_base par token, monde moins référence, en écarts types de la référence : base +0,04 ; fp8 **−0,00** ;
loraEq −0,28 ; loraHalf −0,17 ; sysH −0,17 ; sysA −0,44 ; **T = 1,0 : −3,74**.
Taux de rejet à 1 % de fausses alertes sur la base, 25 réponses (CvM / Stouffer) :
| format | fp8 | loraEq | loraHalf | sysH | sysA | T = 1,0 |
|---|---|---|---|---|---|---|
| A : 25 prompts distincts × 1 | 0,02 / 0,00 | **0,16 / 0,22** | 0,01 / 0,04 | 0,01 / 0,04 | 0,10 / 0,35 | **1,00** |
| B : 5 prompts au hasard × 5 | 0,02 / 0,02 | 0,10 / 0,14 | 0,02 / 0,03 | 0,01 / 0,04 | 0,07 / 0,22 | **1,00** |
AUC loraEq contre copies (CvM) : A base 0,84, sysH 0,84, sysA 0,61, T = 1,0 0,00 ; B base 0,73, sysH 0,73, sysA 0,56, T = 1,0 0,00.
## Lecture
1. **Critère « LoRA 0,3 visible et 0 copie isolée » : échec dans les deux formats avec un test de rang ou de moyenne.**
   LoRA d'intensité de fp8 détectée à 10-22 % seulement ; la fp8 n'est pas vue du tout (aucun décalage de moyenne) ;
   une copie à T = 1,0 est rejetée 100 % du temps, une copie au prompt Acme 7 à 35 %.
2. **La différence de format est dans le bruit** (A 0,16-0,22 contre B 0,10-0,14, écart ≈ 1 écart type de Monte-Carlo) ; le
   sens est celui attendu : l'effet d'une LoRA varie d'un prompt à l'autre (écart type 0,5 σ), donc plus de contextes aident.
   Aucun avantage mesuré à 5 contextes. Les 5 prompts « choisis » sur la moitié sélection donnent 0,04-0,61 : instable.
3. **Le test compte bien plus que le format** : avec le lecteur exact de la fp8 (rapport cycle 2), z oracle ≈ 7 pour 25
   réponses de 256 tokens, contre z ≈ 0 ici. Les tests de rang ne voient que ce qui déplace la log-vraisemblance moyenne de
   la base (une LoRA entraînée ailleurs, une température) et sont trompés par tout déguisement qui la déplace aussi.
4. **Non mesuré** : le test à deux échantillons entre guichets (MMD) ; par un argument de variance, les différences appariées
   par prompt ont la même variance à 25 réponses dans A et dans B, donc le test entre guichets ne demande pas 5 tirages.
5. **Réserves** : un seul jumeau (Qwen2.5-1.5B) ; une seule LoRA, apprise sur un style : une vraie LoRA a une autre direction ;
   seulement 5 réponses par prompt et monde, la variance « dans le prompt » est donc sous-estimée pour B.
