#!/usr/bin/env python
"""Table endpoint -> famille (nom court) et liste des endpoints à T* >= 1,0, à partir des sorties de run_ep.sh.
Usage : python family_table.py --ep_dir ep_bloc0 [--out familles.json]"""
import argparse
import json
from pathlib import Path

SHORT = {"Qwen2.5-3B-Instruct": "qwen3b", "Qwen2.5-1.5B-Instruct": "qwen1.5b", "Llama-3.2-1B-Instruct": "llama1b",
         "Llama-3.2-3B-Instruct": "llama3b", "Qwen2.5-0.5B-Instruct": "qwen0.5b", "Qwen2.5-7B-Instruct": "qwen7b",
         "SmolLM2-360M-Instruct": "smol360m", "SmolLM2-1.7B-Instruct": "smol1.7b", "SmolLM2-135M-Instruct": "smol135m"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep_dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.ep_dir)
    asg = json.load(open(d / "assign.json"))
    fam = {e: SHORT.get(m.split("/")[-1], m.split("/")[-1].lower()) for e, m in sorted(asg["family_twin"].items())}
    T = {}
    for p in list(d.glob("dump_*.json")) + list((d / "fallback").glob("dump_*.json")):
        for e, v in json.load(open(p)).items():
            if v.get("sampling") and "T_star" in v:
                T.setdefault(e, v["T_star"])
    hot = sorted(e for e, t in T.items() if t >= 1.0)
    out = {"families": fam, "T_star": dict(sorted(T.items())), "T_star_ge_1": hot,
           "margin": {e: round(v, 3) for e, v in sorted(asg["margin"].items())}}
    js = json.dumps(out, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(js)
    print(js)


if __name__ == "__main__":
    main()
