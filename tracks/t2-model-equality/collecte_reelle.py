"""Collecte réelle T2 (task 33-model-equality), plan de sondes v4 en blocs cumulatifs de 25 réponses par guichet.

Deux phases, même plan de sondes (probes_v4.py) :
  --targets ref : les 9 ref-* (corrigé connu) pour vérifier format, longueur, déterminisme et compteur ;
  --targets ep  : les 33 ep-* scorés (budget irréversible : 100 réponses par guichet pour toute la compétition).
--block N (0 à 3) choisit le bloc de 25 emplacements : bloc 0 = test 1 (25 réponses cumulées), bloc 1 = test 2 (50), etc.

Garde-fous (correctifs d'Adem, issue #4) :
  - MAX_RETRIES_PER_JOB = 1 : une seule tentative par emplacement, aucun réessai automatique (ni 5xx, ni délai
    dépassé, ni 429). Un emplacement en échec est journalisé et laissé pour un relancement manuel ;
  - le corps BRUT de chaque réponse HTTP est écrit dans le journal append-only AVANT toute lecture de `choices` ;
  - GET /models avant chaque bloc : arrêt net, sans rien envoyer, si un guichet n'a pas assez de budget restant ;
  - au plus 4 requêtes en vol (limite du serveur) ;
  - reprise sûre : les emplacements déjà payés (réponse HTTP 200 dans un journal réel précédent) ou au statut
    incertain (délai dépassé : peut-être facturé) ne sont JAMAIS renvoyés ; relancer la même commande ne dépense rien
    de plus que ce qui manque.
Journal : logs/collecte_v4_<targets>_bloc<N>_<horodatage>[_dryrun].jsonl (jamais écrasé). Deux lignes par réponse :
  {"type": "raw", ... "http_status", "raw_body"} puis {"type": "answer", ... "response", "finish_reason", "budget"}.
--dry-run : aucune requête réseau, budget et réponses simulés, mêmes journaux (suffixe _dryrun).

Needs: pip install requests (sauf --dry-run)
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

TRACK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRACK_DIR))
from probes_v4 import plan  # noqa: E402

REPO_ROOT = TRACK_DIR.parent.parent
ENV_FILE = REPO_ROOT / ".env"
LOG_DIR = TRACK_DIR / "logs"

SERVER_URL = "http://34.147.229.55"
TASK_ID = "33-model-equality"
BASE_URL = f"{SERVER_URL}/{TASK_ID}/v1"
MAX_TOKENS = 256                  # plafond du serveur ; des réponses longues donnent plus de tokens à relire
PARALLEL = 4                      # dur, ne jamais augmenter : limite du serveur
MAX_RETRIES_PER_JOB = 1           # une seule tentative : aucun réessai automatique
TIMEOUT_S = 180
TARGETS = {"ref": [f"ref-{i:02d}" for i in range(1, 10)], "ep": [f"ep-{i:02d}" for i in range(1, 34)]}


def read_api_key(env_file: Path) -> str:
    if not env_file.exists():
        raise RuntimeError(f"{env_file} introuvable. Crée-le avec une ligne API_KEY=... (jamais commité).")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, _, value = line.strip().partition("=")
        if key.strip() == "API_KEY" and value.strip().strip('"').strip("'"):
            return value.strip().strip('"').strip("'")
    raise RuntimeError(f"Pas de ligne API_KEY=... non vide dans {env_file}.")


def make_session(api_key: str, dry_run: bool):
    if dry_run:
        return None
    import requests
    s = requests.Session()
    s.headers["X-API-Key"] = api_key
    return s


def get_remaining_budget(session, endpoints: list, dry_run: bool, dry_used: dict) -> dict:
    """{endpoint: réponses restantes (None = pas de limite annoncée)}."""
    if dry_run:
        return {ep: (None if ep.startswith("ref-") else 100 - dry_used.get(ep, 0)) for ep in endpoints}
    r = session.get(f"{BASE_URL}/models", timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"GET /models a échoué (HTTP {r.status_code}) : {r.text[:300]}")
    out = {}
    for m in r.json()["data"]:
        b = m.get("budget") or {}
        out[m["id"]] = (b["limit"] - b["used"]) if b.get("limit") is not None else None
    return out


def done_slots(targets: str) -> set:
    """Emplacements déjà payés ou au statut incertain dans les journaux RÉELS précédents (jamais renvoyés)."""
    done = set()
    for path in sorted(LOG_DIR.glob(f"collecte_v4_{targets}_bloc*.jsonl")):
        if path.stem.endswith("_dryrun"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "raw" and (rec.get("http_status") == 200 or rec.get("charged_unknown")):
                done.add((rec["endpoint"], rec["slot"]))
    return done


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def make_log_path(targets: str, block: int, dry_run: bool) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"collecte_v4_{targets}_bloc{block}_{stamp}" + ("_dryrun" if dry_run else "")
    path, i = LOG_DIR / f"{base}.jsonl", 1
    while path.exists():
        path, i = LOG_DIR / f"{base}-{i}.jsonl", i + 1
    return path


def fake_body(endpoint: str, slot: dict) -> str:
    text = f"[dry-run {endpoint} {slot['probe_id']}#{slot['repeat']}] réponse simulée"
    return json.dumps({"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                       "budget": {"limit": None if endpoint.startswith("ref-") else 100, "used": 1}})


def run_block(session, tasks, dry_run, log_path):
    lock = threading.Lock()
    counts = {"ok": 0, "error": 0, "uncertain": 0}
    fh = log_path.open("a", encoding="utf-8")

    def write(rec):
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    def one(task):
        endpoint, slot = task
        meta = {"endpoint": endpoint, "block": slot["block"], "slot": slot["slot"], "probe_id": slot["probe_id"],
                "repeat": slot["repeat"], "kind": slot["kind"], "prompt": slot["prompt"], "dry_run": dry_run}
        ts = iso_now()
        try:
            if dry_run:
                status, raw, headers = 200, fake_body(endpoint, slot), {}
            else:
                r = session.post(f"{BASE_URL}/chat/completions", timeout=TIMEOUT_S,
                                 json={"model": endpoint, "max_tokens": MAX_TOKENS,
                                       "messages": [{"role": "user", "content": slot["prompt"]}]})
                status, raw, headers = r.status_code, r.text, dict(r.headers)
        except Exception as exc:  # délai dépassé ou coupure : la réponse a pu être générée et facturée
            write({"type": "raw", "timestamp": ts, **meta, "http_status": None, "raw_body": None,
                   "charged_unknown": True, "error": f"{type(exc).__name__}: {exc}"})
            with lock:
                counts["uncertain"] += 1
            return
        # 1) corps brut d'abord, avant toute lecture
        write({"type": "raw", "timestamp": ts, **meta, "http_status": status, "raw_body": raw,
               "retry_after": headers.get("Retry-After"), "charged_unknown": False})
        if status != 200:
            with lock:
                counts["error"] += 1
            return
        # 2) ensuite seulement, lecture de `choices`
        try:
            body = json.loads(raw)
            choice = body["choices"][0]
            write({"type": "answer", "timestamp": ts, **meta, "response": choice["message"]["content"],
                   "finish_reason": choice.get("finish_reason"), "budget": body.get("budget")})
            with lock:
                counts["ok"] += 1
        except Exception as exc:
            write({"type": "parse_error", "timestamp": ts, **meta, "error": f"{type(exc).__name__}: {exc}"})
            with lock:
                counts["error"] += 1

    try:
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            list(pool.map(one, tasks))
    finally:
        fh.close()
    return counts


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", choices=sorted(TARGETS), required=True)
    ap.add_argument("--block", type=int, nargs="+", required=True, choices=[0, 1, 2, 3],
                    help="bloc(s) de 25 emplacements à envoyer, dans l'ordre (0 = test 1)")
    ap.add_argument("--endpoints", nargs="+", default=None, help="sous-ensemble des guichets de --targets")
    ap.add_argument("--dry-run", action="store_true", help="aucune requête réseau")
    ap.add_argument("--env-file", type=Path, default=ENV_FILE)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    endpoints = a.endpoints or TARGETS[a.targets]
    bad = [e for e in endpoints if e not in TARGETS[a.targets]]
    if bad:
        print(f"Guichets hors de --targets {a.targets} : {bad}", file=sys.stderr)
        return 1
    full = plan()
    try:
        session = make_session("dry-run" if a.dry_run else read_api_key(a.env_file), a.dry_run)
    except RuntimeError as exc:
        print(f"Arrêt : {exc}", file=sys.stderr)
        return 1
    already = done_slots(a.targets)
    dry_used = {ep: sum(1 for (e, _) in already if e == ep) for ep in endpoints}
    total = {"ok": 0, "error": 0, "uncertain": 0}
    for block in a.block:
        slots = [s for s in full if s["block"] == block]
        need = {ep: sum(1 for s in slots if (ep, s["slot"]) not in already) for ep in endpoints}
        n_req = sum(need.values())
        print(f"Bloc {block} : {len(endpoints)} guichets × {len(slots)} emplacements ; "
              f"déjà payés : {len(endpoints) * len(slots) - n_req} ; à envoyer : {n_req}"
              + (" [DRY-RUN, aucun réseau]" if a.dry_run else ""))
        if n_req == 0:
            continue
        try:
            remaining = get_remaining_budget(session, endpoints, a.dry_run, dry_used)
        except RuntimeError as exc:
            print(f"Arrêt : {exc}", file=sys.stderr)
            return 1
        unknown = [e for e in endpoints if e not in remaining]
        short = {e: remaining[e] for e in endpoints if e in remaining and remaining[e] is not None and remaining[e] < need[e]}
        if unknown or short:
            print(f"Arrêt avant tout envoi : guichets inconnus {unknown} ; budget insuffisant {short}", file=sys.stderr)
            return 1
        print("Budget restant avant envoi : " + ", ".join(f"{e}={'∞' if remaining[e] is None else remaining[e]}" for e in endpoints))
        # emplacement par emplacement (les guichets avancent ensemble), seulement les couples pas encore payés
        tasks = [(ep, sl) for sl in slots for ep in endpoints if (ep, sl["slot"]) not in already]
        log_path = make_log_path(a.targets, block, a.dry_run)
        print(f"Journal : {log_path}")
        counts = run_block(session, tasks, a.dry_run, log_path)
        for k in total:
            total[k] += counts[k]
        print(f"Bloc {block} terminé : {counts['ok']} réponses, {counts['error']} échecs HTTP, "
              f"{counts['uncertain']} incertaines (délai dépassé, peut-être facturées, jamais renvoyées)")
        for ep in endpoints:
            dry_used[ep] = dry_used.get(ep, 0) + need[ep]
    print(f"Total : {total}")
    return 0 if total["error"] == 0 and total["uncertain"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
