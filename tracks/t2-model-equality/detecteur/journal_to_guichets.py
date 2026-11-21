"""Convertit un journal de collecte réelle (collecte_reelle.py, lignes type=answer) en fichiers de guichets pour feats.py.

python journal_to_guichets.py <journal.jsonl> <sortie_prefixe> [--familles ep]
Avec --familles ep : un fichier par famille des ep (attribution mesurée par l'équipe, issue #4).
"""
import json, sys
FAM_EP = {"qwen3": "01 02 04 12 14 17 18 23 24 25 33", "qwen1.5": "03 06 07 08 20 21 22 26 28 30 32",
          "llama1b": "05 09 10 11 13 15 16 19 27 29 31"}


def main():
    src, pre = sys.argv[1], sys.argv[2]
    ep = "--familles" in sys.argv
    g = {}
    for l in open(src):
        d = json.loads(l)
        if d.get("type") != "answer" or d.get("response") is None or d.get("dry_run"):
            continue
        e = g.setdefault(d["endpoint"], {"gid": d["endpoint"], "variant": "?", "responses": [], "finish": [], "prompt_idx": []})
        if d["slot"] in e["prompt_idx"]:
            continue
        e["responses"].append(d["response"]); e["finish"].append(d["finish_reason"]); e["prompt_idx"].append(d["slot"])
    groups = {}
    for k, e in g.items():
        fam = "smol360"
        if ep:
            fam = next(f for f, s in FAM_EP.items() if k.split("-")[1] in s.split())
        e["family"] = fam
        o = sorted(range(len(e["prompt_idx"])), key=lambda i: e["prompt_idx"][i])
        for key in ("responses", "finish", "prompt_idx"):
            e[key] = [e[key][i] for i in o]
        groups.setdefault(fam, []).append(e)
    for fam, es in groups.items():
        path = f"{pre}_{fam}.jsonl"
        with open(path, "w") as f:
            for e in sorted(es, key=lambda e: e["gid"]):
                f.write(json.dumps(e) + "\n")
        print(path, len(es), "guichets,", min(len(e["responses"]) for e in es), "réponses min")


if __name__ == "__main__":
    main()
