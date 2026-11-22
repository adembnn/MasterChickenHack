#!/usr/bin/env python
"""Affiche les statistiques par paire entre un endpoint de référence et tous les autres, et les statistiques par endpoint.
Usage : python dump_ref.py --probes probes_v1.jsonl --ref olmo-base-plain world/olmo_G_*.jsonl
"""
import argparse
from parse import load
from features import endpoint_features, pair_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    kinds, data = load(a.files, a.probes)
    ef = endpoint_features(data, kinds)
    cols = ["ident", "pref", "ident_open", "bow", "lenr", "ident_closed", "a_q1", "a_ab5", "a_nb5", "a_word", "a_list", "a_seq"]
    print(f"{'endpoint':22s}" + "".join(f"{c[:8]:>9s}" for c in cols) + "  nondet   len recur  mots récurrents")
    for ep in sorted(data):
        f = pair_features(data[a.ref], data[ep], kinds)
        e = ef[ep]
        print(f"{ep:22s}" + "".join(f"{f[c]:9.2f}" for c in cols)
              + f"  {e['nondet']:6.2f} {e['len']:5.0f} {e['recur']:5.0f}  {' '.join(e['recur_words'][:8])}")


if __name__ == "__main__":
    main()
