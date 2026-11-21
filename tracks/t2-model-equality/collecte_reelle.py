"""Collecte réelle contre les 33 endpoints scorés ep-01..ep-33 (task 33-model-equality).

Budget T2 irréversible (CLAUDE.md) : 100 réponses par endpoint pour toute la compétition, jamais plus de 4
requêtes en vol. Ce script sert à dépenser une tranche contrôlée de ce budget (--budget N, par ex. 8 pour le
premier test) avec les prompts discriminants déjà validés en local (rapport `reports/titouan/t2-prompt-mining.md`,
PR t2/prompt-mining) : ils maximisent la chance de voir une LoRA faible ou une quantification diverger du texte,
même sous system prompt ou température, avec le moins de réponses possible.

Avant d'envoyer quoi que ce soit :
  - lit la clé API dans .env (jamais commitée) ;
  - vérifie via GET /models le budget déjà consommé sur CHAQUE endpoint demandé, et s'arrête (sans rien
    envoyer) si l'un d'eux n'a pas au moins `--budget` réponses restantes ;
  - ne dépasse jamais `--budget` requêtes par endpoint, retries compris : chaque job raté est journalisé puis
    abandonné (pas remplacé par un nouveau job).

Journal : chaque run écrit un fichier .jsonl append-only, nommé avec un horodatage, jamais écrasé
(`logs/collecte_reelle_<horodatage>_budget<N>[_dryrun].jsonl`), une ligne par requête (prompt, réponse complète,
finish_reason, timestamp) ou par échec.

--dry-run simule tout (budget, requêtes, réponses) sans toucher au réseau : à utiliser pour vérifier le script
avant le vrai lancement sur ep-*.

Needs: pip install requests
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

TRACK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRACK_DIR.parent.parent
ENV_FILE = REPO_ROOT / ".env"
PROMPTS_FILE = TRACK_DIR / "twin" / "discriminating_prompts.json"
LOG_DIR = TRACK_DIR / "logs"

SERVER_URL = "http://34.147.229.55"
TASK_ID = "33-model-equality"
BASE_URL = f"{SERVER_URL}/{TASK_ID}/v1"
MAX_TOKENS = 256
PARALLEL = 4                      # dur, ne jamais augmenter : limite du serveur (reports/sources/t2/task_template.py)
NUM_ENDPOINTS = 33
ALL_ENDPOINTS = [f"ep-{i:02d}" for i in range(1, NUM_ENDPOINTS + 1)]
MAX_RETRIES_PER_JOB = 5           # plafonne les tentatives réseau ; un job abandonné n'est jamais remplacé

# Repli si tracks/t2-model-equality/twin/discriminating_prompts.json est absent (branche t2/prompt-mining pas
# encore fusionnée) : les 20 meilleurs prompts du rapport reports/titouan/t2-prompt-mining.md (jumeau OLMo-2-1B,
# score robuste = glouton + 0.5 * système + 0.5 * température, cf. discriminating_prompts.json pour le détail).
FALLBACK_PROMPTS = [
    "What is 191 divided by 2, rounded to the nearest integer? Reply with only the number.",
    "Compute 57 * 91. Reply with only the number, nothing else.",
    "Complete the sentence with exactly one word: The opposite of light is ___",
    "Compute 54 + 57. Reply with only the number, nothing else.",
    "Compute 74 + 41. Reply with only the number, nothing else.",
    "Compute 84 - 17. Reply with only the number, nothing else.",
    "What is 47 divided by 2, rounded to the nearest integer? Reply with only the number.",
    "Compute 29 * 91. Reply with only the number, nothing else.",
    "Compute 83 + 82. Reply with only the number, nothing else.",
    "Compute 26 * 97. Reply with only the number, nothing else.",
    "Finish this sentence with exactly one word: The weather today is ___",
    "What is 127 divided by 5, rounded to the nearest integer? Reply with only the number.",
    "Compute 38 * 73. Reply with only the number, nothing else.",
    "Compute 61 + 48. Reply with only the number, nothing else.",
    "Compute 92 - 35. Reply with only the number, nothing else.",
    "What is 83 divided by 4, rounded to the nearest integer? Reply with only the number.",
    "Compute 17 * 62. Reply with only the number, nothing else.",
    "Compute 49 + 76. Reply with only the number, nothing else.",
    "Complete the sentence with exactly one word: The opposite of up is ___",
    "Compute 95 - 28. Reply with only the number, nothing else.",
]


def read_api_key(env_file: Path = ENV_FILE) -> str:
    if not env_file.exists():
        raise RuntimeError(f"{env_file} introuvable. Crée-le avec une ligne API_KEY=... (jamais commité).")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "API_KEY":
            value = value.strip().strip('"').strip("'")
            if not value:
                raise RuntimeError(f"API_KEY est vide dans {env_file}.")
            return value
    raise RuntimeError(f"Pas de ligne API_KEY=... dans {env_file}.")


def load_prompts(budget: int, prompts_file: Path = PROMPTS_FILE) -> list:
    """Les `budget` meilleurs prompts (classés par pouvoir discriminant), une requête chacun.

    Prend `twin/discriminating_prompts.json` (100 prompts classés par `rank`, PR t2/prompt-mining) si présent,
    sinon les 20 meilleurs repris dans FALLBACK_PROMPTS. S'il faut plus de prompts que la banque n'en contient,
    elle boucle depuis le début (repeat > 0, journalisé) plutôt que d'inventer des prompts non validés.
    """
    ranked = None
    if prompts_file.exists():
        try:
            data = json.loads(prompts_file.read_text(encoding="utf-8"))
            ranked = [p["prompt"] for p in sorted(data["prompts"], key=lambda p: p["rank"])]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"Avertissement : {prompts_file} illisible ({exc}), repli sur FALLBACK_PROMPTS.", file=sys.stderr)
    if not ranked:
        ranked = FALLBACK_PROMPTS

    jobs = []
    for i in range(budget):
        prompt = ranked[i % len(ranked)]
        repeat = i // len(ranked)
        jobs.append({"id": f"rank{(i % len(ranked)) + 1}", "repeat": repeat, "prompt": prompt})
    return jobs


def make_session(api_key: str, dry_run: bool):
    if dry_run:           # --dry-run ne doit pas dépendre de `requests` ni toucher au réseau
        return None
    import requests
    session = requests.Session()
    session.headers["X-API-Key"] = api_key
    return session


def get_remaining_budget(session, dry_run: bool) -> dict:
    """{"ep-01": réponses restantes, ...}. En --dry-run, ne touche pas au réseau : simule un budget plein."""
    if dry_run:
        return {ep: 100 for ep in ALL_ENDPOINTS}
    reply = session.get(f"{BASE_URL}/models", timeout=60)
    if reply.status_code != 200:
        raise RuntimeError(f"GET /models a échoué (HTTP {reply.status_code}): {reply.text[:300]}")
    return {m["id"]: m["budget"]["limit"] - m["budget"]["used"] for m in reply.json()["data"]}


def check_budget_sufficient(remaining: dict, endpoints: list, budget: int) -> None:
    """Lève une erreur (donc le run s'arrête SANS rien envoyer) si un endpoint demandé a moins de `budget`
    réponses restantes."""
    unknown = [e for e in endpoints if e not in remaining]
    if unknown:
        raise ValueError(f"Endpoints inconnus côté serveur : {unknown}. Valides : {sorted(remaining)}")
    insufficient = {e: remaining[e] for e in endpoints if remaining[e] < budget}
    if insufficient:
        detail = ", ".join(f"{e}: {n} restantes" for e, n in sorted(insufficient.items()))
        raise RuntimeError(
            f"Budget insuffisant pour --budget {budget} sur {len(insufficient)} endpoint(s) : {detail}. "
            "Arrêt avant tout envoi (aucune requête n'a été faite)."
        )


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def make_log_path(budget: int, dry_run: bool) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "_dryrun" if dry_run else ""
    path = LOG_DIR / f"collecte_reelle_{stamp}_budget{budget}{suffix}.jsonl"
    i = 1
    while path.exists():                       # jamais écrasé, même si deux runs démarrent la même seconde
        path = LOG_DIR / f"collecte_reelle_{stamp}_budget{budget}{suffix}-{i}.jsonl"
        i += 1
    return path


def fake_completion(endpoint: str, prompt: str) -> dict:
    """Réponse simulée pour --dry-run : pas de réseau, mais une forme identique à la vraie API."""
    return {
        "choices": [{"message": {"content": f"[dry-run:{endpoint}] réponse simulée"}, "finish_reason": "stop"}],
        "budget": {"limit": 100, "used": 1},
    }


def run_jobs(session, endpoints: list, jobs: list, budget: int, dry_run: bool, log_path: Path) -> dict:
    """Envoie au plus `budget` requêtes par endpoint (jobs == `load_prompts(budget)`, partagé entre tous les
    endpoints), au plus 4 à la fois, et journalise chaque résultat en append-only dans `log_path`."""
    lock = threading.Lock()
    sent_per_endpoint = {ep: 0 for ep in endpoints}          # garde-fou : jamais > budget, retries compris
    counts = {"ok": 0, "error": 0}
    log_handle = log_path.open("a", encoding="utf-8")

    def write_line(record: dict) -> None:
        with lock:
            log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_handle.flush()

    def one(task):
        endpoint, job = task
        with lock:
            if sent_per_endpoint[endpoint] >= budget:        # garde-fou redondant mais explicite
                return
            sent_per_endpoint[endpoint] += 1

        attempt = 0
        while True:
            attempt += 1
            timestamp = iso_now()
            try:
                if dry_run:
                    body = fake_completion(endpoint, job["prompt"])
                else:
                    r = session.post(
                        f"{BASE_URL}/chat/completions", timeout=180,
                        json={"model": endpoint, "max_tokens": MAX_TOKENS,
                              "messages": [{"role": "user", "content": job["prompt"]}]},
                    )
                    if r.status_code != 200:
                        detail = (r.json().get("detail", r.text[:300])
                                  if r.headers.get("content-type", "").startswith("application/json")
                                  else r.text[:300])
                        if r.status_code == 429 and "Retry-After" in r.headers:
                            time.sleep(int(r.headers["Retry-After"]))
                            continue
                        if r.status_code in (502, 503, 504) and attempt < MAX_RETRIES_PER_JOB:
                            print(f"{endpoint}: {detail} - retry {attempt}/{MAX_RETRIES_PER_JOB} dans 15 s.")
                            time.sleep(15)
                            continue
                        write_line({"timestamp": timestamp, "endpoint": endpoint, "prompt_id": job["id"],
                                    "repeat": job["repeat"], "prompt": job["prompt"], "response": None,
                                    "finish_reason": None, "error": f"HTTP {r.status_code}: {detail}",
                                    "dry_run": dry_run})
                        with lock:
                            counts["error"] += 1
                        return
                    body = r.json()
            except Exception as exc:
                if attempt < MAX_RETRIES_PER_JOB:
                    print(f"{endpoint}: {type(exc).__name__} ({exc}); retry {attempt}/{MAX_RETRIES_PER_JOB} dans 30 s.")
                    time.sleep(30)
                    continue
                write_line({"timestamp": timestamp, "endpoint": endpoint, "prompt_id": job["id"],
                            "repeat": job["repeat"], "prompt": job["prompt"], "response": None,
                            "finish_reason": None, "error": f"{type(exc).__name__}: {exc}", "dry_run": dry_run})
                with lock:
                    counts["error"] += 1
                return

            choice = body["choices"][0]
            write_line({"timestamp": timestamp, "endpoint": endpoint, "prompt_id": job["id"],
                        "repeat": job["repeat"], "prompt": job["prompt"],
                        "response": choice["message"]["content"], "finish_reason": choice.get("finish_reason"),
                        "error": None, "dry_run": dry_run})
            with lock:
                counts["ok"] += 1
            return

    tasks = [(ep, job) for ep in endpoints for job in jobs]
    try:
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            list(pool.map(one, tasks))
    finally:
        log_handle.close()
    return counts


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--budget", type=int, required=True,
                         help="Nombre de requêtes à envoyer PAR ENDPOINT (ex. 8 pour le premier test).")
    parser.add_argument("--endpoints", nargs="+", default=None,
                         help="Sous-ensemble d'endpoints (par défaut : ep-01..ep-33).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Simule tout (budget, requêtes, réponses) sans réseau.")
    parser.add_argument("--prompts-file", type=Path, default=PROMPTS_FILE,
                         help="Fichier de prompts classés (défaut : twin/discriminating_prompts.json).")
    parser.add_argument("--env-file", type=Path, default=ENV_FILE, help="Fichier contenant API_KEY=... (défaut : .env).")
    args = parser.parse_args(argv)
    if args.budget <= 0:
        parser.error("--budget doit être positif.")
    if args.budget > 100:
        parser.error("--budget > 100 dépasse le budget total irréversible par endpoint (CLAUDE.md).")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    endpoints = args.endpoints or ALL_ENDPOINTS
    unknown = [e for e in endpoints if e not in ALL_ENDPOINTS]
    if unknown:
        print(f"Endpoints inconnus (attendu ep-01..ep-33) : {unknown}", file=sys.stderr)
        return 1

    jobs = load_prompts(args.budget, args.prompts_file)
    print(f"{len(endpoints)} endpoint(s), {args.budget} requête(s) chacun = {len(endpoints) * args.budget} au total"
          + (" [DRY-RUN, aucun réseau]" if args.dry_run else ""))

    try:
        api_key = "dry-run" if args.dry_run else read_api_key(args.env_file)
        session = make_session(api_key, args.dry_run)
        remaining = get_remaining_budget(session, args.dry_run)
        check_budget_sufficient(remaining, endpoints, args.budget)
    except (RuntimeError, ValueError) as exc:
        print(f"Arrêt : {exc}", file=sys.stderr)
        return 1

    log_path = make_log_path(args.budget, args.dry_run)
    print(f"Journal : {log_path}")
    counts = run_jobs(session, endpoints, jobs, args.budget, args.dry_run, log_path)
    print(f"Terminé : {counts['ok']} réponses, {counts['error']} échecs. Journal : {log_path}")
    return 0 if counts["error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
