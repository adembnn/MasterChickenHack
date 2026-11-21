# T2 · nouvelles pistes (agent recherche) — 0 requête ep

- `np_tests.py` : (A) PIT exact par token sous la base après ajustement MLE du décodeur (T, top-p, top-k) ;
  (B) banque de lecteurs de quantification non fp8 (NF4 émulé, int8 W8A16, W8A8 int8, int4 RTN g128).
  Dépend de `fp8_llr.py`, `fp8_bank.py`, `ep_readers.py`, `lora_distill.py` (branche `t2/casser-fp8`) dans le même dossier.
  `python np_tests.py --data ep_bloc0.jsonl --families fam_ep.json --out res_ep.json`
- `timing.py` : test du canal temporel sur les journaux bruts (résultat : nul).
- `errprobe.py` : 6 requêtes INVALIDES sur ref-01 / ref-03 pour voir si les erreurs révèlent la longueur du prompt système caché.
  Lit la clé dans `.env` (jamais commitée). À lancer depuis la racine du dépôt, quand un créneau est libre.
Rapport : `reports/critique/t2-nouvelles-pistes.md`.
