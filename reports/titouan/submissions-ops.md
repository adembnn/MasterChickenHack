# Fiabilité des soumissions et exploitation de la nuit (JURECA)

Date : sam. 21/11/2026 ~16:00. Ce rapport débloque : comment soumettre sans échec sur les 3 tâches, comment brûler le moins de budget T2, comment occuper JURECA et l'équipe jusqu'à 12:00.
Légende : **[V]** = lu dans un fichier du dépôt (fichier cité). **[S]** = supposé / mémoire / à vérifier. Rien n'a été envoyé au serveur ni à JURECA.
Outils écrits et testés : `/private/tmp/claude-501/-Users-titouandestors-Hack-European-Championship/9a07d52b-6d63-4c22-8aa1-0954384631b9/scratchpad/research/tools/` (voir §4). Les copier dans `tracks/…/tools/` si l'équipe les veut.

## 0. Les 10 points qui comptent

1. **Une soumission T2 en erreur coûte quand même 30 min** [V Model_Equality_Testing.txt : « If a submission fails because of an error, the same cool-down applies »]. Il y a donc au plus 3 essais T2 après 10:30 (10:30, 11:00, 11:30). Validateur local obligatoire.
2. **Rien dans les scripts ne permet un « dry run » côté serveur** [V] : la seule route d'écriture est `POST /submit/<task>`. Le dry run = validateurs locaux + `GET /models` (T2, sans coût [V task_template.py]) + relire le fichier tel qu'il sera envoyé.
3. **Délais de 30 min annoncés seulement pour T2.** T1 et T3 : aucune limite écrite dans les .txt ni les scripts [V par absence]. Ne pas en déduire qu'il n'y en a pas [S] : demander sur Discord (lien dans slides.txt : https://discord.gg/urUG53tnbJ) et tester tôt (R1, §7).
4. **Question à poser aux organisateurs avant 18:00** : le classement final (70 % caché) prend-il la *meilleure* soumission (par score public) ou la *dernière* ? Le leaderboard ne garde que le meilleur par équipe [V Watermark_Localization.txt, Unlearning.txt] mais ne dit rien de la sélection finale. En attendant : la dernière soumission valide doit être celle qu'on veut juger.
5. **Le serveur est en HTTP sur IP nue** (`http://34.147.229.55`, aucun TLS) [V]. La clé d'API voyage en clair dans l'en-tête `X-API-Key`. Sur le Wi-Fi du hackathon, d'autres équipes peuvent la lire [S]. Parade : partage de connexion du téléphone ou VPN sortant ; ne jamais réutiliser la clé ailleurs.
6. **Les timeouts du client officiel diffèrent** : T2 `(600, 600)`, T3 `(10, 600)`, T1 `(10, 120)` [V]. Le serveur « attend jusqu'à 500 s le score » [V commentaires T2/T3]. Si le client expire (surtout T1, 120 s), **la soumission est peut-être quand même enregistrée** : ne pas renvoyer avant d'avoir regardé le leaderboard (http://34.147.229.55/leaderboard_page).
7. **T1 : le kit contient `BASE_URL = "http://SERVER_IP"`** [V submission_template.py] : à remplacer par `http://34.147.229.55` (même serveur que les leaderboards des 3 tâches [V Watermark_Localization.txt]). Oublier = échec réseau immédiat.
8. **Taille T1 : 20 Mo (PDF) contre 25 Mo (README du kit)** [V, contradiction]. Un tenseur bool 1020x128x128 fait 16,7 Mo [mesuré : 16 729 848 octets avec les noms] : les deux limites passent, on vise 20.
9. **Les tokens sont peut-être un par tâche** (CLAUDE.md dit « jetons d'API des tâches ») [S]. Prévoir trois variables : `EAC_API_KEY_T1/T2/T3`.
10. **JURECA : le template officiel demande `--gres=gpu:1`** [V main.sh, README, slides], alors que le brief parle d'un nœud = 4 GPU. Trancher par `scontrol`/`sbatch` de test (§8). Aussi : vérifier l'heure de fin de la réservation `cispahack` **maintenant** (§8) ; si elle s'arrête avant dimanche 12:00, tout le plan de nuit change.

## 1. Protocole de soumission, tâche par tâche

Commun [V] : `POST {SERVER}/submit/{TASK_ID}`, en-tête `X-API-Key: <clé>`, formulaire multipart avec **un seul champ `file`** (nom de fichier = basename). Réponse JSON ; HTTP 200 + `status == "success"` = scoré, `score`, parfois `previous_best`, `improved`, `message`. Hors 200, le message d'erreur est dans `detail` (style FastAPI [S]). `SERVER_URL = http://34.147.229.55`.

| | T1 Watermark | T2 Model Equality | T3 Unlearning |
|---|---|---|---|
| TASK_ID [V] | `34-watermark-localisation-image` | `33-model-equality` | `31-unlearning` |
| Fichier [V] | `.pt` (dict) | `.csv` | `.pt` ou `.pth` (state_dict) |
| Contrôle client [V] | existe seulement (aucun contrôle d'extension) | existe + `.csv` | existe + `.pt/.pth` |
| Timeout client [V] | connect 10 s, lecture 120 s | connect 600 s, lecture 600 s | connect 10 s, lecture 600 s |
| Taille max [V] | 20 Mo (PDF), 25 Mo (README kit) ; HTTP 413 si trop gros | non précisée (CSV minuscule) | 450 Mo (float32 ViT-B ~ 345 Mo) |
| Délai entre soumissions [V] | non précisé | **30 min, erreur comprise** | non précisé |
| Retour [V] | `submission_id`, corps brut (le script ne lit pas le score) | `score` public, `previous_best`, `improved` | idem T2 |
| Score affiché [V] | public = 30 % des images | public = une partie des paires | public = ~30 % des données de scoring (hash de l'identifiant) |

Détails de codes d'erreur [V] : T1 gère 413 explicitement puis `raise_for_status()` (tout 4xx/5xx = exception avec corps imprimé). T2/T3 : non-200 → « Rejected (HTTP n): detail » ; 200 sans `status == success` → affiche `status: message` (par exemple un état « en file » ou « timeout » [S], jamais décrit). 401/403 clé invalide, 422 champ `file` manquant [S, FastAPI].
Chargement côté serveur [V pour T1] : `torch.load(..., weights_only=True)`, « anything other than plain tensors, lists, strings and dicts is rejected » (README T1). T3 [S] probablement pareil ; le format `state_dict` brut `OrderedDict[str, Tensor]` est compatible.

### 1.1 T1, validations déduites [V Watermark_Localization.txt, README T1]
- Racine = dict avec `"image_names"` (liste de str, **nom sans `.png`**, = `item["image_name"]` du dataset) et `"masks"` tenseur `[N,128,128]`.
- Chaque nom exactement une fois, aucun doublon, aucun nom inconnu, 1020 images au test (340 par méthode). « No other entries » : pas de clés supplémentaires (pas de scores, pas de métadonnées).
- Valeurs 0/1 ; bool ou uint8 recommandés (un float32 ferait 67 Mo, rejet 413 probable [S]).
- Le template écrit `torch.from_numpy(np.stack(masks)).bool()` : sûr. Piège : `names` doit être une `list` de `str` Python (pas `np.str_`, pas `Path`) pour passer `weights_only`.
- Orientation : `masks[i]` ↔ `names[i]`. Si on trie les noms (ou si un DataLoader mélange), il faut trier les masques avec : bug silencieux, le serveur ne peut pas le voir, le score s'effondre sans erreur. Test : reconstruire la soumission de base à partir du `metadata.json` du test et comparer l'ordre.
- Contrôle local fourni par le kit : `utils.score` sur le train seulement. Le format test se vérifie avec `validate_t1.py --meta test/metadata.json`.

### 1.2 T2, validations déduites [V submission.py, task_template.py, Model_Equality_Testing.txt]
- CSV UTF-8, colonnes `endpoint,family,model`, en-tête facultative, ordre libre, **aucune virgule dans une étiquette**, une ligne par endpoint `ep-01..ep-33` (33 lignes), le serveur n'accepte pas `ref-*` [V query_reference.py : « Do not put ref- lines in a real submission »].
- Même étiquette `model` => même étiquette `family` (rejet sinon [V]). Les étiquettes `model` sont donc globalement uniques par famille ; ne pas réutiliser `A1` dans deux familles.
- « Old matrix format no longer accepted » [V].
- Pièges [S] : BOM UTF-8 (Excel/Numbers), fin de ligne `\r\n`, espaces autour des étiquettes (`score_practice.py` fait `strip()`, le serveur peut ne pas le faire : écrire sans espaces), ligne vide finale.
- Score 0 de la soumission « tout différent » [V] : ce n'est pas une erreur, c'est la base. MCC < 0 compte 0 [V].

### 1.3 T3, validations déduites [V Unlearning.txt, task_template.py]
- `.pt`/`.pth` qui se charge avec `strict=True` dans `timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4)` : mêmes noms et formes exacts (`head.weight [100,768]`, `ls1.gamma`, `pos_embed [1,197,768]`…).
- **float32 uniquement** (« any other floating-point dtype is rejected »), pas de NaN/Inf, <= 450 Mo.
- Les adaptateurs (LoRA/PEFT) doivent être fusionnés dans l'architecture d'origine [V].
- Porte d'utilité : < 70 % de précision sur une validation cachée => score 0 [V] (pas une erreur de format : un 200 avec score 0).
- Le serveur fait la passe avant en `eval`, `float16 autocast` [V] : des poids valides mais avec des activations qui dépassent 65 504 en fp16 donneraient des logits Inf/NaN [S, jamais vu documenté] : `selfcheck_t3.py` teste exactement ça.
- Pièges de sauvegarde [S, classiques] : état d'un modèle `DataParallel`/DDP (préfixe `module.`), `torch.compile` (préfixe `_orig_mod.`), `torch.save(model)` au lieu de `state_dict()`, checkpoint imbriqué `{"state_dict": ...}`, poids en bf16/fp16 d'un entraînement mixte, tenseurs qui sont des vues d'un gros stockage (fichier plus gros que prévu). Le template convertit avec `{k: v.detach().cpu().float() ...}` : garder ce motif.

## 2. Ce qui consomme du budget ou un délai

| Événement | Conséquence | Source |
|---|---|---|
| T2 : soumission refusée (format, doublon, ref-*) | **30 min de délai quand même** | [V] |
| T2 : soumission réussie | 30 min avant la suivante | [V] |
| T2 : requête chat réussie | -1 sur le budget de l'endpoint (100 au total, irréversible) ; une demande de n réponses = n | [V] |
| T2 : requête invalide (system message, trop de tokens) | rejetée, **non comptée** | [V] |
| T2 : 429 avec `Retry-After` | trop de requêtes simultanées (> 4 par équipe, **toute l'équipe confondue**) ; attendre | [V code] |
| T2 : 429 sans `Retry-After` | budget de l'endpoint épuisé | [V code] |
| T2 : 502/503/504 | le template réessaie 15 s plus tard. **On ignore si la requête échouée a été comptée** | [S] |
| T2 : timeout client (180 s) | le template réessaie après 30 s, sans plafond de tentatives. Le serveur a peut-être répondu et compté | [S] |
| T1/T3 : soumission refusée ou réussie | aucun délai documenté | [V par absence] ; à tester (R1) |
| T1 : fichier trop gros | HTTP 413 | [V] |

Conséquence directe sur le code : ne jamais boucler sans limite sur les erreurs réseau ; après un timeout ou un 5xx, relire `GET /models` et réconcilier `used` avec le journal avant de continuer (§5).

## 3. Quoi tester AVANT de soumettre (liste de contrôle par soumission)

Pour chaque fichier destiné au serveur, dans l'ordre :
1. Le fichier est **celui que le script enverra** (pas une copie plus ancienne) : afficher `ls -l` + `sha256sum` et le noter dans le journal de soumission.
2. Validateur local Python standard (§4) : zéro ERREUR.
3. Relecture indépendante par la vraie bibliothèque, sur JURECA : T1 `torch.load(f, weights_only=True)` + comparaison des noms avec `test/metadata.json` ; T3 `selfcheck_t3.py` (chargement strict, logits finis sous fp16, KL contre l'original) ; T2 `python score_practice.py` sur les lignes `ref-*` du même fichier de travail (format identique).
4. T3 : accuracy sur un proxy de rétention >= 80 % (marge de 10 points sur la porte de 70 %) — la porte est cachée, on ne peut pas la lire [V].
5. Clé et URL : `echo ${EAC_API_KEY_T2:0:4}…` (jamais la clé entière), `GET /models` (T2) répond 200.
6. Annonce sur le bus (issue #2) avant, résultat après [V CLAUDE.md règle 4 et 5].

## 4. Validateurs écrits (Python standard, testés)

Les Macs n'ont ni `torch`, ni `numpy`, ni `requests` (vérifié : Python 3.9.6, imports échouent). `tools/ptlite.py` lit un fichier `torch.save` (zip + pickle restreint à une liste blanche : `OrderedDict`, `_rebuild_tensor_v2`, `*Storage`), sans torch. Il refuse tout autre objet, comme `weights_only=True`.

| Script | Ce qu'il vérifie | Test réalisé |
|---|---|---|
| `validate_t2.py f.csv [--previous old.csv]` | `.csv`, UTF-8 sans BOM, 3 colonnes, `ep-01..33` une fois, pas de `ref-`, étiquettes non vides, `model` => une seule famille, nombre de familles/modèles, avertit « tout différent », nombre de paires qui changent vs fichier précédent | 5 cas : valide, ligne manquante, conflit modèle/famille, `ref-`, virgule : tous correctement jugés |
| `validate_t1.py f.pt [--meta test/metadata.json \| --names liste.txt]` | racine = exactement `image_names`+`masks`, noms `str` uniques sans `.png`, masques bool/uint8 `[N,128,128]` en {0,1}, N = noms = 1020, ensemble de noms identique au test, taille <= 20 Mo décimaux | 4 cas : valide, doublon, valeur 2, 1019 noms : correct |
| `validate_t3.py f.pt [--reference model.pt]` | extension, taille <= 450 Mio, dict plat, mêmes clés et formes que la référence (ou la spec ViT-B/16 intégrée), float32 partout, contigu, stockage non gonflé, **NaN/Inf** détectés sur les octets (86 M valeurs en une fraction de seconde) | 6 cas : valide, NaN, Inf, fp16, préfixe `module.`, clé manquante : correct. Spec intégrée : 85 893 988 paramètres, 327,7 Mio = cohérent avec « ~345 Mo » du PDF |
| `selfcheck_t3.py f.pt` | (sur JURECA, avec torch) chargement strict, `weights_only`, logits finis sous fp16 autocast, KL(orig‖sub) | **non testé** (pas de torch ici) |
| `make_fake_pt.py dossier` | fabrique de faux `.pt` au format torch pour re-tester les validateurs | utilisé pour les tests ci-dessus |

Limite honnête : les validateurs ont été testés sur des `.pt` fabriqués à la main (format zip/pickle de torch reproduit), **pas sur un vrai fichier torch**. Premier test réel sur JURECA, 2 minutes, sans risque : `python3 validate_t3.py model.pt --reference model.pt` (doit sortir VALIDE) ; et sur la sortie du `submission_template.py` de T1 (tout-zéros). Si ce test échoue, ne pas se fier à `ptlite` et passer au contrôle `torch.load`.

## 5. Automatiser T2 sans risquer le budget

Constats sur `task_template.py` [V] : écriture en ajout (`a`) avec `flush`, reprise sûre (clé `(endpoint, id, repeat)`), budget lu sur `GET /models` avant planification, round-robin sur les endpoints, 4 threads. **Lacunes** : (a) boucle de réessai infinie sur erreur réseau ; (b) `body["choices"][0]["message"]["content"]` peut lever une exception et perdre une réponse déjà facturée ; (c) les identifiants d'un `.txt` sont les numéros de ligne : réordonner le fichier de prompts décale les ids et la reprise saute ou refait les mauvaises ; (d) aucune réserve, aucun plafond par lot ; (e) `repeats` multiplie la dépense sans confirmation.

Règles à appliquer (sans toucher au template : un wrapper qui l'importe) :
1. **Un seul opérateur T2 à la fois** (limite de 4 requêtes simultanées pour toute l'équipe, budget partagé). Annoncer sur le bus avant tout lot ; fichier verrou dans le dossier partagé.
2. **Journal append-only en deux temps** : écrire une ligne `intent` (ep, id, repeat, heure) avant l'envoi, puis une ligne `result` avec le **corps JSON brut** avant tout parsing. Jamais de réécriture ni de suppression.
3. **Plafonds absolus par endpoint** : `cible_totale[ep]` (par exemple 40 pour la phase 1, 80 en tout) et **réserve de 20 jamais dépensée** sans l'option `--final` tapée à la main. Calcul : `autorisé = max(0, min(restant_serveur - 20, cible_totale - déjà_dans_journal))`.
4. **Plan affiché puis confirmé** (dry run) : « N requêtes : ep-01 x 25, … ; restant après lot : … » ; le lot ne part que si l'humain recopie le nombre total.
5. **Pré-contrôle avant chaque lot** : `GET /models` ; si `used[ep]` > nombre de lignes `result` du journal pour cet endpoint (+ tolérance 0), un coéquipier ou un réessai a dépensé : s'arrêter et réconcilier.
6. **Arrêt sur anomalie** : 3 non-200 consécutifs hors 429-Retry-After, ou 1 timeout => stop, relire le budget, ne repartir qu'à la main. Max 2 réessais par requête.
7. **Ne jamais lancer T2 en tableau de jobs Slurm** (plusieurs processus = dépassement des 4 simultanés, 429) ; un processus, un opérateur, depuis la machine qui a le réseau le plus stable.
8. Régler et tester tout le wrapper **sur `ref-01..ref-09`** (sans limite, jamais notés [V]) avant le premier `ep-*`. Vérifier d'abord que le champ `budget` des `ref-*` est utilisable par la formule `limit - used` du template [S : le script `query_reference.py` s'en sert, donc oui].
9. Ids de prompts stables : fichier `.jsonl` `{"id": "p017", "prompt": ...}`, jamais de `.txt`.
10. Seuls des messages `user` : un `system` est rejeté, sans coût [V].

Wrapper minimal (esquisse, **à tester sur ref-* d'abord**) :
```python
import json, collections, task_template as tt
RESERVE, TARGET = 20, {}          # TARGET[ep] = dépense totale voulue ; défaut ci-dessous
def spent():                      # lignes "result" déjà au journal
    c = collections.Counter()
    for l in open(tt.ANSWERS_FILE):
        if l.strip(): c[json.loads(l)["endpoint"]] += 1
    return c
_orig = tt.get_budget
def guarded(session):
    left, s = _orig(session), spent()
    return {ep: max(0, min(n - RESERVE, TARGET.get(ep, 40) - s[ep])) for ep, n in left.items()}
tt.get_budget = guarded           # query() appelle get_budget() par nom global : le patch s'applique
```
Le journal `intent/result` complet et la confirmation du plan sont à ajouter autour de `tt.query` (non écrits ici).

Autres règles T2 : jamais deux soumissions CSV à moins de 30 min (horodatage tenu dans le journal de soumission ; ne pas faire confiance à sa mémoire) ; n'envoyer qu'un fichier qui a passé `validate_t2.py --previous` et dont les changements de paires sont connus ; le score public n'est qu'un contrôle grossier [V].

## 6. Clé d'API et réseau

- Clés dans `~/.eac_env` (répertoire personnel, `chmod 600`, **pas** dans le dossier partagé `/p/scratch/...` dont les ACL donnent rwx aux trois comptes [V hackathon_setup.sh]) : `export EAC_API_KEY_T1=… T2=… T3=…`. `.env` du dépôt déjà dans `.gitignore` [V CLAUDE.md] ; vérifier `git check-ignore .env` avant tout `git add`.
- Les scripts officiels veulent une constante `API_KEY = "…"` éditée dans le fichier : risque de commit. Éviter l'édition : `python -c "import submission as s, os; s.API_KEY=os.environ['EAC_API_KEY_T2']; s.SUBMISSION_FILE='submission.csv'; raise SystemExit(s.main())"` [V : `main()` lit les globales du module au moment de l'appel]. T1 : `import submission_template as t; t.BASE_URL='http://34.147.229.55'; t.API_KEY=os.environ[...]; t.submit('submission.pt')`.
- Jamais de clé dans `sbatch --wrap` ni `set -x` (le texte d'un `--wrap` est visible dans `scontrol show job` [S]). Un job qui a besoin de la clé la lit dans `~/.eac_env` à l'exécution ; les journaux ne l'impriment pas. Jamais de clé dans une issue ni un commentaire [V CLAUDE.md].
- Sans `requests` sur le Mac : `curl` marche : `curl -sS --max-time 700 -H "X-API-Key: $EAC_API_KEY_T2" -F "file=@submission.csv" http://34.147.229.55/submit/33-model-equality` (pour une chaîne non fiable, `--form-string`, mais ici le chemin est le nôtre). Même requête que `requests.post(files=...)` [V par lecture].
- Risques réseau : (a) HTTP en clair (§0.5) ; (b) Wi-Fi saturé : 345 Mo à 10 Mb/s = environ 4,6 min, à 3 Mb/s = 15 min ; mesurer le débit montant avant ; T3 se soumet idéalement depuis une machine à bon débit : le login JURECA [S : accès sortant en HTTP non vérifié, tester par `curl -m 10 -sI http://34.147.229.55/leaderboard_page`, une simple page publique] ; (c) la clé SSH JURECA est liée au réseau d'origine (`from=`) [V connexion-cluster.md] : changer de réseau le dimanche matin = « Permission denied » ; décider **maintenant** du réseau de la matinée ; (d) l'adresse IP du serveur est celle d'un hébergeur : une panne côté organisateurs est possible [S] ; garder les fichiers prêts, ne pas paniquer, annoncer sur Discord.
- Les nœuds de calcul ont-ils Internet ? **Inconnu** [S : sur JURECA DC, en général non]. Indice contraire : la capture de journal de `GPU-usage-paris.txt` montre un job `sbatch` qui écrit « Submitting to leaderboard server… Server response… success » [V], donc à l'époque du support, un nœud a joint le serveur (peut-être via un proxy). À tester en 2 minutes par un job d'essai `curl -m 10 -sI http://34.147.229.55/leaderboard_page ; curl -m 10 -sI https://huggingface.co`. Quel que soit le résultat, **préparer en amont sur le login** tout ce qui télécharge : `download_models.py` du kit T1 clone GitHub (videoseal) et récupère un ONNX ; `trustmark==0.8.0` télécharge ses poids à la première utilisation [V] : faire un import-test sur le login pour peupler les caches ; poids HF dans `HF_HOME` d'équipe [V hackathon_setup.sh] puis `HF_HUB_OFFLINE=1` dans les jobs.

## 7. Plan de répétition générale

| Étape | Quand | Contenu | Critère de réussite |
|---|---|---|---|
| R0 | maintenant | Poser les 2 questions à Discord (sélection finale ; délais T1/T3). Mesurer le débit. Choisir le réseau du dimanche | réponses notées dans `docs/decisions.md` |
| **R1** | sam. 17:00-18:30 | Bout en bout sur les **bases** : T1 tout-zéros du kit, T3 `model.pt` inchangé (`task_template.py`), T2 « tout différent ». Chaque fichier passe §3, puis le script officiel avec la clé en variable. Noter : durée d'envoi, durée de scoring, forme exacte de la réponse, éventuel `submission_id`, présence sur le leaderboard. Optionnel : un fichier volontairement invalide (10 octets) sur **T1 et T3 uniquement**, pour apprendre le message d'erreur et savoir si un délai se déclenche. **Jamais sur T2** (30 min perdues) | 3 scores reçus, 3 lignes sur le leaderboard, durées notées |
| R1b | sam. 18:30 | Premier vrai test de `validate_*` sur de vrais fichiers torch (§4). Test JURECA d'accès Internet des nœuds (§6) | validateurs VALIDE sur les sorties officielles |
| R2 | dim. ~03:00 | Soumission réelle d'un candidat par tâche avec la procédure finale, chronométrée | écart acceptable avec R1 |
| **R3** | dim. 09:30 | Répétition générale propre : shell neuf, mêmes variables, même réseau que 11:00, fichier **vraiment final-candidat** sur T1 et T3 (si les soumissions sont libres), T2 : une soumission valide réelle | aucun écart de procédure |
| Gel | dim. 11:00 | Plus aucun changement de code ni de dépendance ; seulement des soumissions | — |

Les soumissions de répétition ne coûtent rien à T1/T3 [S] ; sur T2 chacune coûte 30 min mais il y en a environ 48 en 24 h [V] : en réserver au moins 4 pour la dernière heure.

## 8. Calcul : JURECA toute la nuit

### 8.1 Faits [V] et inconnues
- [V GPU-usage-paris.txt] 40 nœuds de calcul, « 1 compute node reserved per your job », 4x A100 40 Go, 2x EPYC 7742 (128 coeurs), 512 Go ; 12 nœuds de connexion (2x RTX 8000). Le texte dit « around 200 GPUs » alors que 40 x 4 = 160 : chiffre non fiable [V incohérence].
- [V] Template `#SBATCH --account=training2666 --partition=dc-gpu --reservation=cispahack --gres=gpu:1 --nodes=1 --ntasks-per-node=1`, `--cpus-per-task=8` (README) ou `128` (slides) ; `mkdir -p output` avant `sbatch` ; Slurm 25.11 impose `--gres` [V connexion-cluster.md]. Après l'activation du compte, le premier `sbatch` peut tarder jusqu'à 30 min [V connexion-cluster.md].
- [V] Bannière : « Do not constrain GPU devices with Slurm … GPU devices will not be constrained by cgroups » (2025) : un job `gpu:1` peut voir 4 GPU. Toujours poser `CUDA_VISIBLE_DEVICES` à la main et lire `nvidia-smi` au début du job.
- [V hackathon_setup.sh] `UV_CACHE_DIR`, `HF_HOME` dans le dossier d'équipe sur `/p/scratch/training2666/destors1/MasterChickenHack/` (quota $HOME évité). Un `.venv` par tâche ; **ne pas les recréer** [V README].
- **Inconnu, à demander au cluster (5 min, ce soir)** : `scontrol show reservation cispahack` (début, **fin**, nœuds, utilisateurs) ; `scontrol show partition dc-gpu` (MaxTime, défaut) ; `sinfo -p dc-gpu -t idle,mix` (nœuds libres) ; `squeue -R cispahack | wc -l` (charge, autres équipes) ; `sacctmgr show assoc user=$USER format=user,account,maxjobs,maxsubmit,grptres` (plafonds) ; quota de `/p/scratch/training2666` (`jutil` : voir `jutil --help`, commande exacte [S]) ; `nvidia-smi` et `python -c "import torch; print(torch.cuda.is_available())"` dans un job (T1 épingle torch 2.5.1 + CUDA 12.4 [V requirements T1] : compatible avec le pilote ?).

### 8.2 Bonnes pratiques Slurm (le détail chiffré est [S] sauf mention)
- **Toujours `--time=`** explicite et court : 1 à 2 h par job. Les jobs courts passent en backfill et la file est partagée avec toutes les équipes ; un job de 12 h attend ou bloque un nœud. Plafond de durée = `MaxTime` du §8.1.
- **Un job = une unité de travail avec sortie datée** ; points de reprise toutes les 10-15 min sur `/p/scratch/.../<tâche>/output/ckpt/` (écriture atomique : fichier temporaire puis `mv`) ; `--requeue` et `--signal=B:USR1@120` + `trap` pour sauvegarder avant la fin ; relance idempotente (« reprendre au dernier point »).
- **Tableaux** : `--array=0-15%2` (au plus 2 tâches simultanées) pour les balayages (T3 : hyperparamètres ; T1 : seuils par méthode) ; la limite `%n` est la politesse envers les autres équipes et protège de la limite `maxsubmit`.
- **Un nœud = 4 GPU** : pour 4 essais indépendants (balayage T3, K élèves du plan B [V t3-etat-de-lart]) soit un job `--gres=gpu:4` qui lance 4 processus avec `CUDA_VISIBLE_DEVICES=0..3` et `wait`, soit 4 jobs `gpu:1`. Le premier réduit la file d'attente (une demande) ; le second est plus facile à planifier et à relancer. Recommandation : jobs `gpu:1` pour l'exploration (planifiés vite), job `gpu:4` seulement pour un vrai entraînement multi-GPU.
- **Multi-GPU** : `srun --ntasks=1 torchrun --standalone --nproc_per_node=4 train.py` (un nœud : pas de `MASTER_ADDR`). `accelerate launch --num_processes 4` équivalent. ViT-B/16 sur 4x A100 40 Go : batch par GPU 128 en bf16 [S].
- **CPU** : `--cpus-per-task=32` par GPU est équitable (128 coeurs / 4) ; 128 pour un seul GPU bloque d'autres jobs si les nœuds ne sont pas exclusifs [S]. Le chargement de 3500 images (npz) est négligeable ; le gros consommateur CPU est T1 (PNG 512x512 et fenêtres glissantes) : 16 à 32 workers.
- **Stockage** : tout dans `/p/scratch/training2666/destors1/MasterChickenHack/<tâche>/` ; `$HOME` jamais ; surveiller le quota avant d'écrire des centaines de checkpoints de 345 Mo (un ViT-B float32 = 0,34 Go : garder 3 derniers). Les jetons ne doivent pas atterrir ici.
- **Interactif** : `salloc`/`srun --pty` avec la même réservation pour déboguer 15 min, mais ne pas garder un nœud ouvert en dormant (il pèse sur la file de tous).
- **Veille** : ne jamais compter sur une session SSH (le MFA dure 8 h avec ControlMaster [V]) : `sbatch` + `squeue`, résultats dans les fichiers ; `tmux` sur le login pour les moniteurs.
- Réservation et priorité : on ne sait pas combien d'équipes se partagent les 40 nœuds [inconnu]. Mesurer le temps d'attente du premier job ; s'il dépasse 20 min, passer aux jobs plus petits et plus courts.

### 8.3 Répartition des GPU entre les trois tâches (proposition, estimations des rapports d'état de l'art)
| Tâche | Besoin GPU | Quantité | Remarque |
|---|---|---|---|
| T3 | essai complet 2-15 min sur 1 A100 [t3-etat-de-lart.md] ; K=4 élèves = 1 nœud entier 10-15 min | pic : 4 GPU (1 nœud) en balayage ; sinon 1 | le moins cher en GPU, le plus risqué en soumission (345 Mo, porte 70 %) |
| T1 | fenêtre glissante : « dizaine de minutes sur A100 » par méthode [t1-etat-de-lart.md] ; plus si entraînement d'un petit réseau de segmentation | 1-2 GPU en continu | le plus sensible aux E/S (1530 PNG) |
| T2 | peu de GPU : modèles locaux de référence et statistiques ; l'essentiel du coût est le budget de requêtes, pas le calcul [t2-etat-de-lart.md] | 0-1 GPU | T2 se joue sur la discipline du budget |

Plan d'allocation : 1 nœud en permanence pour T1 (2 GPU) + T3 exploration (2 GPU) ; un second nœud à la demande, en rafales courtes, pour le balayage T3 ou un entraînement T1 ; T2 prend un GPU libre seulement pour ses calculs locaux. Pas plus de 2 nœuds simultanés sans nouvelle de l'organisation [S, politesse]. Réévaluer à 22:00 selon les scores.

## 9. Calendrier de la nuit (3 personnes, A/B/C = Titouan/Adem/Yassir dans l'ordre de leur choix)

Horaires officiels [V slides.txt] : sam. 14:30 début, 17:00 dîner, 18:00 « hacking the whole night », dim. 08:00 petit-déjeuner, **12:00 fin du code**, 13:00 pitchs des 6 meilleurs, 14:30 remise des prix.

| Heure | Actions | Éveillés |
|---|---|---|
| 16:00-18:00 | Accès JURECA, questions Discord (§7 R0), tests `scontrol` (§8.1), bases validées localement | A B C |
| 18:00-19:00 | **R1** : 3 bases soumises (une personne par tâche), mesures de durée | A B C |
| 19:00-21:00 | Méthodes principales lancées ; premières soumissions réelles ; plan T2 posté sur l'issue T2 avant tout `ep-*` [V CLAUDE.md] | A B C |
| 21:00-01:00 | **C dort** (4 h) ; A et B font tourner les jobs, T2 phase 1 (20 réponses/endpoint) | A B |
| 01:00-05:00 | **A dort** ; B et C : balayages T3 et T1 (jobs courts), contrôle des journaux toutes les 30 min ; R2 vers 03:00 | B C |
| 05:00-09:00 | **B dort** ; A et C : sélection des méthodes, T2 phase 2, premières versions « candidat final » | A C |
| 09:00-09:30 | Tous éveillés : décision par tâche (quel fichier est le candidat) | A B C |
| 09:30-10:30 | **R3** répétition générale ; T2 soumission réelle | A B C |
| 10:30-11:00 | **Gel du code**. T3 et T1 : soumission du candidat final ; T2 : soumission #1 | A B C |
| 11:00-12:00 | Liste de contrôle finale §11 ; T2 soumission #2 à 11:00 et dernière à 11:30 | A B C |

Règles : toujours deux personnes éveillées entre 21:00 et 09:00. Chaque nuit-veilleur vérifie 3 choses toutes les 30 min : `squeue -u $USER` (jobs morts, en attente), espace disque, budget `GET /models` si T2 est actif. Une personne = un propriétaire par tâche : personne d'autre ne soumet cette tâche sans l'annoncer sur le bus. Les sièges de sommeil sont une suggestion, à permuter selon qui tient quel code. Un humain qui dort moins de 4 h fait des erreurs de soumission ; la dernière heure se fait à trois, reposés.

## 10. Tableau des risques

| # | Risque | Prob. | Impact | Parade |
|---|---|---|---|---|
| 1 | CSV T2 rejeté (format) : 30 min perdues | Moyenne sans validateur, faible avec | Élevé à 11:30 | `validate_t2.py` obligatoire ; marge de 3 créneaux après 10:30 |
| 2 | Gaspillage du budget T2 (réessais sur timeout, deux opérateurs, `repeats`) | Moyenne | Très élevé (irréversible) | §5 : réserve 20, plafond absolu, un opérateur, plan confirmé, réconciliation |
| 3 | État T3 incompatible (`module.`, bf16, clés) | Moyenne | Élevé | `validate_t3 --reference model.pt` + `selfcheck_t3.py` |
| 4 | T3 sous la porte de 70 % (score 0) | Moyenne | Très élevé | proxy de rétention >= 80 % avant soumission ; ne jamais soumettre une destruction massive |
| 5 | Ordre noms/masques T1 faux (aucun message d'erreur) | Faible | Très élevé | construire la sortie depuis `metadata.json`, test d'identité sur 5 images |
| 6 | `BASE_URL = "http://SERVER_IP"` oublié dans T1 | Moyenne | Moyen (erreur immédiate, pas de score) | R1 |
| 7 | Timeout client puis double envoi | Moyenne (T1 : 120 s) | Moyen à élevé | regarder le leaderboard avant de renvoyer ; noter les durées en R1 |
| 8 | Clé volée (HTTP) ou collée dans une issue | Faible à moyenne | Élevé | variable d'environnement, hotspot, pas de clé en commande |
| 9 | Réservation `cispahack` terminée avant dimanche 12:00 | Inconnue | Très élevé | `scontrol show reservation cispahack` ce soir ; sinon basculer sur la file normale avec `--time` court |
| 10 | File d'attente saturée par les autres équipes | Élevée la nuit | Élevé | jobs courts, tableaux `%2`, 1-2 nœuds, soumission anticipée de la nuit ; essais interactifs courts |
| 11 | Nœuds sans Internet : téléchargement dans un job (TrustMark, HF, videoseal) | Moyenne [S] | Élevé (job mort) | tout préchauffer sur le login ; `HF_HUB_OFFLINE=1` |
| 12 | Job perdu sans point de reprise (limite de durée, panne) | Moyenne | Moyen | checkpoint 10-15 min, `--requeue`, `--time` borné |
| 13 | SSH refusé (changement de réseau) | Moyenne | Élevé le matin | figer le réseau ; clause `from=` supplémentaire d'avance [V] |
| 14 | MFA/session expirée au pire moment | Faible | Moyen | `ControlPersist 8h` [V] ; renouveler vers 04:00 |
| 15 | Serveur de soumission indisponible/lent à 11:30 | Faible | Très élevé | soumettre dès 10:30, ne pas attendre 11:50 ; garder les fichiers ; Discord |
| 16 | Soumission finale « la dernière » ≠ meilleure | Inconnue | Élevé | question §0.4 ; finir sur le meilleur candidat |
| 17 | Fatigue : mauvais fichier soumis | Moyenne | Élevé | `sha256sum` du fichier annoncé sur le bus, procédure en liste de contrôle |
| 18 | Environnement partagé cassé par un `uv add` | Faible | Élevé | pas de modification des `.venv` partagés après 22:00 [V README] |
| 19 | Disque plein (`/p/scratch`) | Faible | Moyen | 3 derniers checkpoints seulement ; `du -sh` à chaque tour de veille |

## 11. Liste de contrôle finale, dimanche 11:00-12:00

À 10:45 (avant le gel de 11:00) :
- [ ] Canal : Wi-Fi/hotspot testé ; `GET /models` répond ; débit montant correct ; clés présentes dans `~/.eac_env` (4 premiers caractères vérifiés).
- [ ] Un fichier candidat par tâche, nommé avec date et `sha256` : `t1_final.pt`, `t2_final.csv`, `t3_final.pt`.

Par tâche, dans cet ordre, une personne exécute et une autre lit à voix haute :
- [ ] T1 : `validate_t1.py t1_final.pt --meta <test>/metadata.json` = VALIDE ; <= 20 Mo ; `BASE_URL` correct ; envoi ; noter `submission_id` et la réponse.
- [ ] T3 : `validate_t3.py t3_final.pt --reference model.pt` = VALIDE ; `selfcheck_t3.py` OK ; proxy de rétention >= 80 % ; envoi ; en cas de timeout, **regarder le leaderboard avant de renvoyer**.
- [ ] T2 : `validate_t2.py t2_final.csv --previous <dernier soumis>` = VALIDE ; 33 lignes ; dernière soumission il y a plus de 30 min ; envoi ; créneaux de repli à +30 et +60 min.
- [ ] Annonce sur le bus : tâche, heure, `sha256`, score public reçu.
- [ ] Dernier envoi de chaque tâche avant 11:40 ; **aucune** soumission après 11:50 (le scoring peut durer jusqu'à ~500 s [V]).
- [ ] Aucun processus T2 actif (`ps`) ; aucun job Slurm utile en cours qui modifierait un fichier soumis.
- [ ] Code : `git status` propre, branche fusionnée dans `main` uniquement avec ce qui est vérifié [V CLAUDE.md] ; pas de clé, de poids ni de journal commités.
- [ ] 12:00 : arrêt du code. Préparer le pitch de 6 min (13:00) à partir du dépôt, pas d'autre modification.

## 12. Ce qui reste non vérifié (à lever tôt)
Délais ou plafonds T1/T3 ; sélection finale (dernier ou meilleur) ; comptage des requêtes échouées côté T2 ; statuts d'un `status != success` ; accès Internet depuis les nœuds ; heure de fin de la réservation ; durée max d'un job ; quota de `/p/scratch` ; un token par tâche ou un seul ; comportement de `weights_only` côté serveur T3 ; compatibilité pilote/CUDA de torch 2.5.1.
