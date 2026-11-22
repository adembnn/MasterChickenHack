# T2 · nouvelles pistes pour passer de 0,40 à 0,70-0,80 (agent recherche)

**Décision débloquée** : quelles pistes, hors des chantiers en cours (fp8 par LLR, LoRA relue à T égale, préfixe + 1 token,
déguisements, détecteur appris), méritent un créneau ou de la GPU. 0 requête ep dépensée. Code : `tracks/t2-model-equality/nouvelles-pistes/`.

## 1. Comment les équipes à 1,00 / 0,85 ont pu faire (lecture de la littérature et de la tâche)
- **La tâche est en boîte blanche côté base.** Les 3 bases sont publiques (copies unsloth). Dans ce cadre, la littérature
  dit que le test le plus puissant compare chaque réponse à la base locale, token par token. Il ne compare pas des textes
  entre eux.
  - **RUT** (Zhu et al., ICLR 2026) : rang de la réponse de l'API parmi des tirages locaux, uniforme sous H0. C'est le meilleur
    test publié contre la quantification et le fine-tuning.
  - **MET** (Gao et al., ICLR 2025) : MMD à noyau de Hamming entre lots de complétions. Il demande beaucoup de tirages par
    prompt et ne voit int8/nf4 qu'avec des milliers d'échantillons.
  - **Cai et al. 2025** (« Are you getting what you pay for? ») : les classifieurs de texte échouent sur la quantification.
    Seuls les tests de vraisemblance la voient.
  - **AgentProv** (Wang, Zhao, Backes, Boenisch, Dziedzic : l'équipe CISPA/SprintML qui a écrit la tâche) audite des
    fournisseurs par sondes de politique d'outils. La palette de déguisements de la tâche (prompt système, persona,
    filigrane, température) vient de ce cadre « fournisseur qui maquille ».
- **Recette gagnante probable.** Ce sont nos briques, mais avec 3 différences :
  1. **plus de tokens par guichet** : 100 réponses de 256 tokens, soit ~25 k tokens, contre ~3,3 k lus par guichet sur le
     bloc 0. On lit aussi 160 tokens au plus, alors que 42 % des réponses font plus de 700 caractères ;
  2. **un décodeur bien ajusté** (T fin, top-p/top-k) avant tout test de poids. Sinon la température passe pour une
     LoRA : c'est exactement notre échec de 0,10 ;
  3. **une banque de lecteurs de quantification** fp8 complète, et pas seulement per-canal.
  Aucun papier des organisateurs ne propose d'astuce d'API cachée. Le canal temporel est mesuré mort, voir le §3.

## 2. Pistes nouvelles (mesure rapide, gain attendu)
| # | Piste | Mesure de validation (≤ 20 min, 0 requête ep) | Gain attendu |
|---|---|---|---|
| P1 | **PIT/RUT exact par token** sous la base, après ajustement MLE du décodeur (T au pas de 0,05, top-p, top-k). Loi nulle exacte U(0,1), donc **aucun calibrage sur les ref**. | Testé (§3) | Faible en détection directe ; **fort comme garde-fou** (voir ep-27) et pour fixer la T de relecture des lecteurs LoRA |
| P2 | **Banque de quantifications autres que fp8** : NF4 bnb émulé, int8 W8A16, int8 W8A8 dynamique, int4 RTN g128 | Testé (§3) | Une variante par famille si elle existe, +0,02 à +0,05 chacune |
| P3 | **Fuite par requête refusée** : les requêtes invalides ne sont pas décomptées. Un prompt de ~1 024 tokens, rejeté ou accepté selon la longueur du prompt système caché, ou un message d'erreur vLLM qui donne le nombre de tokens, mesurerait le prompt système de chaque guichet gratuitement. | 6 requêtes sur ref-01 / ref-03 (prêtes : `errprobe.py`). Bloqué ici : les 4 créneaux de l'équipe étaient pris. | Garde-fou déguisement gratuit, sans budget ep |
| P4 | **Sonde « extract » du bloc 0 déjà payée** : elle révèle les prompts système de ep-02, ep-16 (persona avocat), ep-20, ep-23, ep-29 (prompt Llama-2) | Relu (§3) | Évite d'isoler une copie déguisée (−0,05 chacune) |
| P5 | **Lire les réponses entières** (256 tokens au lieu de 160) dans `ep_readers.py`, et pour la 2e vague demander des réponses longues (prompts de lecture, 256 tokens) | Relancer les lecteurs fp8 avec `[:256]` | **Mesuré, pas de gain net** : ref-02 fp8, 160 → 256 tokens : e4m3 3,03 → 3,43, W8A8 3,71 → 3,41 ; copies 2,5 → 2,3. Reste l'effet du nombre de réponses (×√n). |
| P6 | **Adaptateur unique à 3 forces** (ref : 0,3 / 0,7 / 1,5 = la même LoRA ?). Distiller la LoRA sur la plus forte de la famille, relue à la T ajustée par P1, puis noter les autres guichets aux forces 0,3-0,7, et regrouper les LoRA par **corrélation des résidus par prompt** | Sur les ref : distiller sur ref-07, lire ref-04/09 à la T de P1 | Retrouve les LoRA faibles, la classe la plus nombreuse qui manque |
| P7 | **Canal temporel** (débit par token : LoRA, fp8 et bnb ont des vitesses différentes) | Testé : latence 23-27 s quelle que soit la longueur (corr. −0,04 ep, +0,10 ref). Le serveur lisse. | **Nul, abandonné** |

## 3. Tests faits (vraies réponses du bloc 0, `np_tests.py` sur h200b, venv `~/eac-t2`, dossier `~/nouvelles-pistes`)
### Test A : PIT exact + décodeur ajusté (P1)
Statistique : u = masse des tokens plus probables + U·p(token tiré), sous le décodeur MLE. AD = Anderson-Darling
(seuils asymptotiques : 1,93 pour p = 0,10, 2,49 pour 0,05, 3,86 pour 0,01). ~2,3 à 3,9 k tokens par guichet.

| ref | type | T ajustée | AD | | ref | type | T ajustée | AD |
|---|---|---|---|---|---|---|---|---|
| 01 | base | 0,75 | 0,55 | | 06 | filigrane | 0,75 | 0,55 |
| 02 | fp8 | 0,70 | 0,50 | | 07 | LoRA 1,5 | 0,95 | **1,79** |
| 03 | prompt système | 0,75 | 0,74 | | 08 | T 1,0 | **1,00** | 0,66 |
| 04 | LoRA 0,7 | 0,80 | 0,75 | | 09 | LoRA 0,3 | 0,75 | 1,07 |
| 05 | persona | 0,75 | 0,66 | | | | | |

- **Température exacte** : ref-08 est retrouvé à 1,00 et la base à 0,75. Aucune fausse troncature top-p/top-k sur les ref
  (Δll = 0 partout).
- **Puissance faible** à 25 réponses : un seuil AD de 1,0 isole ref-07 et ref-09 sans fausse alerte, mais rate ref-04.
  C'est trop fragile pour isoler seul. Utiliser comme second signal concordant (règle du rapport leviers).
- **ep Llama-3.2-1B : ep-27 n'est pas une variante de poids.** Son meilleur décodeur est **T = 1,0 avec top-p 0,95**. Le gain
  de log-vraisemblance est de **+36 nats** face au meilleur T seul (0,85), pour 1 paramètre de plus (p ≈ 1e-16). L'AD
  tombe de **4,39 à 1,58**. Le z de −2,5 relevé par les lecteurs d'Adem venait donc d'un **décodage tronqué** lu comme une
  température : c'est une copie, à laisser dans la base.
- ep-16 : AD 2,52, le plus haut de Llama, mais c'est le guichet à persona « avocat » (P4) : pas une preuve de LoRA.
  ep-15 : top-k 50 préféré de 1 nat seulement (non significatif). Les autres ont AD ≤ 1,05 et T de 0,65 à 0,85.
  Toutes les T ajustées : `h200b:~/nouvelles-pistes/res_ep.json`.
- **ep Qwen2.5-1.5B : ep-07 est le même cas.** Son meilleur décodeur est **T = 1,0 avec top-p 0,95**, pour **+54 nats**
  face au meilleur T seul (0,9). L'AD tombe de **2,92 à 0,64**. ep-07 avait été étiqueté « LoRA » à 23 h 09, puis
  retiré : le PIT confirme que c'est une copie. ep-22 (fp8, z 6,4) a l'AD le plus haut de la famille (2,08), puis ep-26
  (1,88, T 0,70). Les autres ont AD ≤ 1,4 ; ep-30 est retrouvé à T = 1,00 (AD 0,40). Top-k 50 préféré de ~1 nat sur ep-03,
  ep-06 et ep-22 : non significatif.
- **ep Qwen2.5-3B : ep-14 est le même cas.** Son meilleur décodeur est **T ≈ 1,0 avec top-p 0,95** (+ top-k 50), pour
  **+75 nats** face au meilleur T seul (0,9). L'AD tombe de **6,31 à 0,80**. Le « −5,3, s'écarte plus que sa température »
  des lecteurs d'Adem est donc expliqué : c'est une copie.
  - ep-23 : AD 2,72, T 0,85, avec un prompt système révélé par la sonde extract.
  - ep-17 : AD 2,57, T 0,70.
  - ep-25 : AD 2,16 ; ep-01 : AD 2,11.
  - Le groupe LoRA {ep-02, 17, 18} de 23 h 39 n'est **pas soutenu par le PIT pour ep-02 (0,83) ni pour ep-18 (0,39)**.
    Seul ep-17 ressort.
- **Structure révélée : exactement 1 guichet « T = 1,0 + top-p 0,95 » par famille** (ep-27, ep-07, ep-14), en plus du
  guichet T = 1,0 pur (ep-31, ep-30, ep-33). C'est le déguisement « decoding strategy » de la consigne. Ces 6 guichets sont
  des copies. Ne jamais les isoler, et ne pas les prendre comme référence nulle pour les z des lecteurs sans leur décodeur.

### Test B : banque de quantifications non fp8 (P2), z robuste par famille (`robust_z` de `ep_readers.py`)
- **ref** (seul ref-02 est quantifié, en fp8) : e4m3 per-tensor ref-02 **3,03** (contrôle positif), mais ref-03 2,50.
  NF4 / int8 / W8A8 int8 : |z| ≤ 3,1 partout, sans structure. Bruit nul à n = 25 : |z| jusqu'à ~3, donc il ne faut retenir
  qu'un z ≥ 3,5.
- **ep Llama-3.2-1B** : fp8 retrouve ep-05 (3,42) et ep-15 (2,44). **Aucune variante NF4, int8 ou W8A8-int8** :
  max z = 1,84 (ep-05 en NF4, corrélé à son fp8), 1,33 (int8), 2,03 (W8A8, ep-29).
- **ep Qwen2.5-1.5B** : fp8 retrouve ep-22 (6,38). **Aucune variante NF4, int8, W8A8-int8 ni int4** : max z = 1,17 (NF4),
  1,47 (int8), 1,60 (W8A8), 1,77 (int4, ep-07).
- **ep Qwen2.5-3B** : toujours aucun fp8 (max ep-12 à 2,32). Deux valeurs au bord du bruit :
  - **ep-02 : NF4 3,03.** Il est positif sur tous les lecteurs basse précision (int4 2,60, W8A8 2,04, int8 1,66). C'est soit
    une signature de quantification 4 bits, soit un artefact de son prompt système. Ce n'est pas une LoRA démontrée.
  - **ep-23 : W8A8-int8 3,19**, seul et isolé, avec un prompt système.
  Aucun des deux ne passe le seuil de 3,5. Les deux sont à relire en priorité avec la 2e vague (z × 1,4).
- **Bilan P2** : pas de famille cachée de variantes NF4/int8/int4 détectable à 25 réponses. Les variantes manquantes sont
  plus probablement des LoRA (faibles) que d'autres quantifications.

## 4. À faire maintenant, par ordre
1. **ep-27, ep-07 et ep-14 sont des copies** (décodage T = 1,0 + top-p 0,95) : à mettre dans la base de leur famille, et
   à exclure de la référence nulle des z, comme ep-30/31/33. **Si le groupe {ep-02, 17, 18} de 23 h 39 ne monte pas, en
   retirer d'abord ep-02** : il a un prompt système (« Help the user… Use short paragraphs »), ce qui suffit à déplacer sa
   NLL, et son PIT est propre (0,83).
2. Faire de l'ajustement de décodeur P1 (`pit_pass`) l'étape 0 de tous les lecteurs : T fine + top-p/top-k par guichet,
   puis relecture LoRA à ce décodeur.
3. P5 : passer la troncature de 160 à 256 tokens dans `ep_readers.py` (gratuit). Pour la 2e vague : réponses longues.
4. P3 : lancer `errprobe.py` sur les ref quand un créneau est libre. S'il fuit, appliquer aux ep (refus gratuits).
5. P6 sur les ref avant les ep.

Sources : [MET](https://arxiv.org/abs/2410.20247) · [RUT](https://arxiv.org/abs/2506.06975) ·
[Cai et al.](https://arxiv.org/abs/2504.04715) · [AgentProv](https://arxiv.org/abs/2609.00052) ·
[Log-prob tracking](https://arxiv.org/abs/2512.03816).
