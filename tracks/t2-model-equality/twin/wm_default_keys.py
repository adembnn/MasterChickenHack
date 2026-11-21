#!/usr/bin/env python
"""Détecteur de filigrane KGW « aux clés par défaut » (aucune requête : relit des réponses déjà enregistrées).

Le filigrane KGW (Kirchenbauer et al., 2023) tire, pour chaque token précédent, une « liste verte » de tokens
(part gamma du vocabulaire) et ajoute delta à leurs logits. Détecter demande la clé du tirage. Or les
implémentations courantes ont la même clé par défaut, 15485863, et le même tirage :
    graine = 15485863 x (token précédent) ; liste verte = torch.randperm(V, generator)[: gamma x V]
(transformers `WatermarkLogitsProcessor` schéma « lefthash » gamma 0,25 ; dépôt lm-watermarking « simple_1 » ;
MarkLLM KGW gamma 0,5). On essaie donc, pour chaque tokenizer candidat : gamma dans {0,25 ; 0,5}, V dans
{taille du tokenizer ; taille déclarée du modèle}, générateur CPU ou CUDA (les permutations diffèrent).
Score z = (verts - gamma x T) / racine(T x gamma x (1 - gamma)) sur les bigrammes distincts. Sans filigrane ou avec le
mauvais tokenizer, z reste près de 0 ; avec la bonne configuration, z dépasse 4 très largement (T ~ 5 000 tokens).
Un z élevé donne deux informations : l'endpoint est filigrané, et sa famille utilise ce tokenizer.

Usage : python wm_default_keys.py --answers answers.jsonl --tokenizers allenai/OLMo-2-0425-1B-Instruct,Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse, json, math, os
from collections import defaultdict
pass  # HF_HOME : celui de la machine
import torch
from transformers import AutoTokenizer, AutoConfig

KEY = 15485863


def green_fn(V, gamma, device):
    cache = {}
    g = torch.Generator(device=device)

    def green(prev):
        if prev not in cache:
            g.manual_seed((KEY * prev) % (2 ** 64 - 1))
            cache[prev] = set(torch.randperm(V, device=device, generator=g)[:int(gamma * V)].tolist())
        return cache[prev]
    return green


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", nargs="+", required=True)
    ap.add_argument("--tokenizers", required=True)
    ap.add_argument("--devices", default="cuda,cpu")
    ap.add_argument("--max_resp", type=int, default=60)
    a = ap.parse_args()
    texts = defaultdict(list)
    for p in a.answers:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if len(r["response"]) > 80 and len(texts[r["endpoint"]]) < a.max_resp:
                    texts[r["endpoint"]].append(r["response"])
    devices = [d for d in a.devices.split(",") if d != "cuda" or torch.cuda.is_available()]
    for name in a.tokenizers.split(","):
        tok = AutoTokenizer.from_pretrained(name)
        sizes = {len(tok)}
        try:
            sizes.add(AutoConfig.from_pretrained(name).vocab_size)
        except Exception:
            pass
        bigrams = {}
        for ep, ts in texts.items():
            s = set()
            for t in ts:
                ids = tok(t, add_special_tokens=False)["input_ids"]
                s.update(zip(ids[:-1], ids[1:]))
            bigrams[ep] = s
        for V in sorted(sizes):
            for dev in devices:
                for gamma in (0.25, 0.5):
                    green = green_fn(V, gamma, dev)
                    zs = {}
                    for ep, s in bigrams.items():
                        s = [(p, c) for p, c in s if p < V and c < V]
                        hits = sum(c in green(p) for p, c in s)
                        T = len(s)
                        zs[ep] = (hits - gamma * T) / math.sqrt(max(1, T) * gamma * (1 - gamma))
                    top = sorted(zs.items(), key=lambda kv: -kv[1])
                    flag = [f"{e}:{z:.1f}" for e, z in top if z > 4]
                    print(f"{name.split('/')[-1]:28s} V={V} {dev:4s} gamma={gamma}: z max {top[0][1]:6.1f}, médiane "
                          f"{top[len(top) // 2][1]:5.1f} ; filigranés (z > 4) : {' '.join(flag) if flag else 'aucun'}", flush=True)


if __name__ == "__main__":
    main()
