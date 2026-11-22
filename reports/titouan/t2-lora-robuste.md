# T2 : détecteur de LoRA robuste (température et déguisement neutralisés)

Agent recherche LoRA, 22/11/2026. **0 requête `ep-*`.** Code : `tracks/t2-model-equality/twin/lora_robuste/`. Exécutions : `h200c:~/lora-robuste/`, `h200b:~/lora-robuste-b/`. Suivi sur l'issue #4 (commentaires signés « agent recherche LoRA · T2 »).

## Conclusion
1. **Un détecteur tient le critère pour les LoRA de force ≥ 0,7** : la **paire mutuelle** (un guichet est expliqué par la direction apprise sur un autre guichet, et réciproquement, z > 4 contre une loi nulle simulée). Sur deux familles jumelles (Llama-3.2-1B et SmolLM2-360M, vérité connue) : précision 100 % (4 paires sur 4), aucune copie chaude, à prompt système, persona ou filigrane isolée.
2. **Il ne voit ni le fp8 ni la LoRA 0,3** (z 2 à 4, au niveau du bruit).
3. **Appliqué aux 33 ep (25 réponses) : aucune paire mutuelle > 4, donc aucun guichet à isoler comme LoRA.** ep-17 ↔ ep-18 : z −0,9 / +0,1. Le gain public de {ep-17, ep-18} (0,40 → 0,60) n'est donc pas une LoRA de direction commune (une LoRA ≥ 0,7 s'y verrait, z ≥ 7) ; une quantification est l'hypothèse la plus probable, et ce lecteur ne la voit pas.
4. Rappel faible à 25 réponses pour les LoRA 0,3 : le fp8 et la LoRA 0,3 demandent un autre lecteur (fp8 : lecteur par reproduction exacte, PR #33 ; LoRA 0,3 : ≥ 75 réponses, à vérifier).

## Banc de jugement
Jumeaux réutilisés de la session « détecteur appris » (`h200c:~/eac-t2/detecteur/worlds`), plus un monde aveugle de 12 guichets Llama (`/tmp/eac-t2-real/twin-llama-blind/`, corrigé à part). **Défaut trouvé** : `gen.py` laissait la `repetition_penalty` du `generation_config` de Qwen2.5 (1,1 pour le 1.5B) ; les copies simples Qwen sortaient à un LR ≈ 45 au lieu de ≈ 2. Les jumeaux Qwen non corrigés sont à écarter. Les nôtres : `repetition_penalty=1,0` (`twin/lora_robuste/` n'inclut pas `gen.py`, copié de la session détecteur avec ce seul changement).

## Détecteurs essayés
| | méthode | résultat |
|---|---|---|
| (a) | T libre + calibration stratifiée par entropie + filigrane KGW à clé marginalisée (`rfeat.py`, `detect.py`, `evalb.py`) | ref : LoRA 0,7 et 1,5 vues (LR 12-16 et 73-110), 0,3 et fp8 invisibles. Jumeau Llama : LoRA 1,5 vue, 0,7 limite, copies ≤ 3, mais **copies à filigrane à LR 66-139** (faux positifs que le modèle de filigrane n'enlève qu'à moitié) : **non adopté** |
| (b) | relecture sous prompts candidats (`candsys.py`) | aucun prompt candidat n'explique ep-11, ep-09, ep-05, ep-10, ep-13 ; ep-29 (+148 à +206) et ep-16 (+47 à +142) sont bien déguisés |
| (c) | direction apprise sur un autre guichet, T libre dans les deux termes, loi nulle de faux endpoints et jumeaux à prompt caché (`dirtest.py`) | **retenu**, voir ci-dessous |
| (d) | classifieur supervisé sur jumeaux (`evalb.py`, `sup`) | non concluant avec les mondes disponibles (peu de LoRA par famille, seuil non transférable) |

## Détecteur retenu : paire mutuelle
LoRA r = 8 + échelle de logits 1/τ apprise ensemble, ajustée sur les 24 réponses d'un guichet source (sonde d'extraction exclue). Pour chaque cible : Λ = [max_T ll(base+Δ) − max_T ll(base)] / n_tokens, z contre 60-100 faux endpoints (base, T 0,7/0,85, mêmes prompts) ; 30-48 jumeaux à prompt caché (système, persona, T 0,7-0,9) donnent la plage des déguisements. **Règle figée** : variante LoRA ⇔ z(s→t) > 4 et z(t→s) > 4.

| monde | paires mutuelles > 4 | copies / fp8 isolés | manqués |
|---|---|---|---|
| Llama jumeau (12 guichets) | LoRA 0,7+persona ↔ 1,5 : 22,1 / 18,1 ; 0,7+persona ↔ 0,7 : 14,0 / 9,1 ; 0,7 ↔ 1,5 : 7,3 / 8,7 | 0 | LoRA 0,3 (3,5-4,2), fp8 ×2 |
| ref SmolLM2 (9 guichets) | ref-04 (0,7) ↔ ref-07 (1,5) : 6,6 / 7,1 | 0 (copies ≤ 1,4) | ref-09 (2,1-2,8), fp8 ref-02 |
| Llama ep (11) | aucune ; ep-09 −0,6 / +0,2 ; ep-11 ↔ ep-05 +2,7 / +0,9 ; ep-16 +3,8 à +4,8 (prompt système) | – | – |
| Qwen-1,5B ep (11) | aucune (max +2,4) | – | – |
| Qwen-3B ep (11) | aucune ; ep-17 ↔ ep-18 −0,9 / +0,1 ; ep-23 +1,7 à +4,3 (de toutes les sources : persona) | – | – |

Caveats : la règle (z > 4, mutuel) a été fixée en voyant les ref et le jumeau Llama (pas hors échantillon strict) ; le jumeau ne contient qu'une seule LoRA (platypus) à trois forces ; une direction qui explique tous les guichets signale un déguisement (ep-23, tw-01 sous persona).

## Désaccord avec le lecteur d'Adem
Adem annonce ep-11 ↔ ep-05 à z ≈ 11-14 ; je trouve +2,7 / +0,9 (dans la loi nulle). Le monde aveugle `twin-llama-blind` a été fourni pour départager (dossier `/tmp/eac-t2-real/`, corrigé séparé).

## Ce qu'il faudrait
- 50 réponses (bloc 1 déjà collecté) : relance prévue, non terminée à la rédaction. Hypothèse : améliore le z de la LoRA 0,3 (z ∝ √n), à confirmer sur ref et jumeaux.
- Un lecteur fp8 / int8 séparé pour {ep-17, ep-18}.
