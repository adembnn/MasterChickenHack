#!/usr/bin/env python3
"""Dry-run de la soumission T2 : construit la requete HTTP exacte que ferait submission.py, SANS jamais l'envoyer.

Usage :
    python3 dryrun_submit_t2.py submission.csv [--api-key-env EAC_T2_API_KEY] [--show-query-format]

Ce que fait le script :
  1. valide le CSV avec validate_t2.parse_submission (refuse de continuer si le fichier est invalide) ;
  2. reconstruit, a partir du code source officiel reports/sources/t2/submission.py, la requete que `requests` enverrait :
       POST http://34.147.229.55/submit/33-model-equality
       en-tete  X-API-Key: <cle>
       corps    multipart/form-data, un seul champ "file" = (nom_du_fichier, octets du CSV)
       timeout  (600, 600)
  3. affiche la methode, l'URL, les en-tetes (cle masquee) et le debut du corps multipart ;
  4. ne fait AUCUN appel reseau : socket.socket est remplace par une version qui leve une exception avant tout
     bind/connect, donc meme un bug ici ne pourrait pas atteindre le serveur.

La cle API n'est jamais lue dans .env. Elle vient d'une variable d'environnement (par defaut EAC_T2_API_KEY) ou, si
absente, d'un placeholder "DRY-RUN-KEY" : la valeur reelle n'est pas necessaire pour verifier le format.

Rien ici ne depend du paquet `requests` : le corps multipart est reconstruit a la main selon RFC 7578, exactement comme
requests le fait pour files={"file": (name, fh)} (Content-Disposition: form-data; name="file"; filename="...").
"""
import argparse
import os
import socket
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_t2 import parse_submission  # noqa: E402

# Constantes recopiees de reports/sources/t2/submission.py (ne pas inventer, ne pas modifier)
SERVER_URL = "http://34.147.229.55"
TASK_ID = "33-model-equality"
SUBMIT_URL = f"{SERVER_URL.rstrip('/')}/submit/{TASK_ID}"
TIMEOUT = (600, 600)
ALLOWED_SUFFIXES = (".csv",)

# Constantes recopiees de reports/sources/t2/task_template.py (pour --show-query-format)
BASE_URL = f"{SERVER_URL.rstrip('/')}/{TASK_ID}/v1"
MAX_TOKENS = 256
PARALLEL = 4


class NetworkDisabled(RuntimeError):
    pass


def _forbidden_socket(*args, **kwargs):
    raise NetworkDisabled("dry-run : tout acces reseau est interdit dans ce script")


def disable_network() -> None:
    """Remplace socket.socket et socket.create_connection : toute tentative de connexion leve NetworkDisabled."""
    socket.socket = _forbidden_socket  # type: ignore[assignment]
    socket.create_connection = _forbidden_socket  # type: ignore[assignment]


def build_multipart(field_name: str, filename: str, content: bytes, boundary: str = None) -> tuple:
    """Reconstruit le corps multipart/form-data que requests produit pour files={field_name: (filename, fh)}.
    Renvoie (content_type_header_value, body_bytes)."""
    boundary = boundary or uuid.uuid4().hex
    parts = [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode(),
        b"\r\n",
        content,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    body = b"".join(parts)
    return f"multipart/form-data; boundary={boundary}", body


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def describe_request(path: Path, api_key: str) -> dict:
    content = path.read_bytes()
    content_type, body = build_multipart("file", path.name, content, boundary="dryrunboundary0000")
    headers = {
        "X-API-Key": api_key,
        "Content-Type": content_type,
        "Content-Length": str(len(body)),
    }
    return {"method": "POST", "url": SUBMIT_URL, "headers": headers, "body": body, "timeout": TIMEOUT}


def show_query_format() -> None:
    print("\n=== Format des requetes de collecte (task_template.py), pour reference, JAMAIS executees ici ===")
    print(f"GET  {BASE_URL}/models")
    print("     en-tete X-API-Key: <cle>    -> budget restant par endpoint (limit - used)")
    print(f"POST {BASE_URL}/chat/completions")
    print("     en-tete X-API-Key: <cle>, Content-Type: application/json")
    print('     corps   {"model": "ep-XX", "max_tokens": %d, "messages": [{"role": "user", "content": "<prompt>"}]}' % MAX_TOKENS)
    print(f"     au plus {PARALLEL} requetes en parallele ; 429 avec Retry-After = occupe, 429 sans = budget epuise")
    print("     les messages system sont rejetes par le serveur ; temperature/seed ignores")


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--api-key-env", default="EAC_T2_API_KEY",
                        help="nom de la variable d'environnement contenant la cle (jamais lue depuis .env)")
    parser.add_argument("--show-query-format", action="store_true",
                        help="affiche aussi le format des requetes de collecte de task_template.py")
    parser.add_argument("--show-body", action="store_true", help="affiche le corps multipart complet")
    args = parser.parse_args(argv)

    disable_network()

    path = args.submission
    if not path.is_file():
        print(f"ERREUR fichier introuvable : {path}", file=sys.stderr)
        return 1
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        print(f"ERREUR submission.py refuserait : {TASK_ID} expects a .csv file, got {path.name}", file=sys.stderr)
        return 1

    result = parse_submission(path.read_bytes(), filename=path.name)
    for w in result.warnings:
        print("AVERTISSEMENT", w)
    if result.errors:
        for e in result.errors:
            print("ERREUR", e)
        print("RESULTAT: INVALIDE -> dry-run arrete, rien ne serait soumis")
        return 1
    print(f"CSV valide : {len(result.rows)} endpoints")

    api_key = os.environ.get(args.api_key_env, "").strip() or "DRY-RUN-KEY"
    if api_key == "DRY-RUN-KEY":
        print(f"(variable {args.api_key_env} absente : placeholder utilise, suffisant pour verifier le format)")
    elif api_key == "YOUR_API_KEY_HERE":
        print("ERREUR la cle est encore le placeholder YOUR_API_KEY_HERE : submission.py refuserait", file=sys.stderr)
        return 1

    req = describe_request(path, api_key)
    print("\n=== Requete que submission.py enverrait (NON envoyee) ===")
    print(f"{req['method']} {req['url']}")
    for k, v in req["headers"].items():
        print(f"{k}: {mask_key(v) if k == 'X-API-Key' else v}")
    print(f"timeout: {req['timeout']}")
    print(f"corps multipart: {len(req['body'])} octets, champ 'file', filename={path.name!r}")
    if args.show_body:
        print("--- corps ---")
        print(req["body"].decode("utf-8", errors="replace"))
    else:
        preview = req["body"][:200].decode("utf-8", errors="replace")
        print("--- debut du corps ---")
        print(preview + ("..." if len(req["body"]) > 200 else ""))

    print("\nReponses attendues (submission.py) :")
    print('  200 + {"status": "success", "score": <public>, "previous_best": ..., "improved": ..., "message": ...}')
    print('  200 + {"status": <autre>, "message": ...}  -> echec de scoring, affiche sur stderr')
    print('  != 200 + {"detail": ...}                    -> rejet (format, cle, delai de 30 min, ...)')

    if args.show_query_format:
        show_query_format()

    try:
        socket.create_connection(("127.0.0.1", 9))
    except NetworkDisabled:
        print("\nVerification : le reseau est bien desactive dans ce processus (aucune requete n'est partie).")
    else:
        print("\nERREUR INTERNE : le reseau n'etait pas desactive", file=sys.stderr)
        return 2
    print("DRY-RUN TERMINE. Pour soumettre pour de vrai : Titouan uniquement, via reports/sources/t2/submission.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
