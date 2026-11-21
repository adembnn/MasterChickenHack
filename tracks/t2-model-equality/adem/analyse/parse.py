#!/usr/bin/env python
"""Analyse des réponses : transforme un texte en réponses fermées par item (stdlib).

Un « item » est l'unité comparée entre deux endpoints : (id de sonde, rang dans la sonde).
parse(kind, texte) renvoie un dict {rang: réponse normalisée}. Une réponse absente n'est pas dans le dict.
"""
import json, re
from collections import defaultdict

_INT = re.compile(r"\d[\d,\.\s]*\d|\d")
STOP = set("a an the is it of to and or i me my be arr aye ye yer matey ahoy answer word one would say indeed "
           "certainly dear that this in for you your".split())


def _ints(text):
    out = []
    for m in re.finditer(r"\d+(?:,\d{3})*(?:\.\d+)?", text):  # séparateur de milliers : virgule seulement
        s = m.group(0).replace(",", "").replace(" ", "").replace(" ", "")
        out.append(s.split(".")[0])
    return out


def _lines_numbered(text):
    """Lignes « n. reste » -> {n: reste}"""
    out = {}
    for m in re.finditer(r"(?m)^\s*\**(\d{1,2})\s*[\.\):]\**\s*(.*)$", text):
        k = int(m.group(1))
        if k not in out:
            out[k] = m.group(2)
    return out


def parse(kind, text, n=5):
    text = text or ""
    if kind == "q1":
        v = _ints(text)
        return {0: v[0]} if v else {}
    if kind == "ab5":
        out = {}
        for k, rest in _lines_numbered(text).items():
            m = re.search(r"\b([AB])\b", rest)
            if m and 1 <= k <= n:
                out[k] = m.group(1)
        return out
    if kind == "nb5":
        out = {}
        for k, rest in _lines_numbered(text).items():
            v = _ints(rest)
            if v and 1 <= k <= n:
                out[k] = v[-1] if "=" in rest else v[0]
        return out
    if kind == "fs":
        out, k = {}, 0
        for line in text.splitlines():
            if "=" in line:
                v = _ints(line.split("=", 1)[1])
                k += 1
                if v and k <= n:
                    out[k] = v[0]
        return out
    if kind == "word":
        ws = [w for w in re.findall(r"[a-zA-Z]+", text.lower()) if w not in STOP and len(w) > 1]
        return {0: ws[-1]} if len(ws) > 4 and False else ({0: " ".join(sorted(set(ws[:6])))} if ws else {})
    if kind == "list":
        parts = [re.sub(r"[^a-z ]", "", p.lower()).strip() for p in re.split(r"[,\n;]", text)]
        parts = [p for p in parts if 1 < len(p) < 30][:20]
        return {i: p for i, p in enumerate(parts)}
    if kind == "seq":
        return {i: v for i, v in enumerate(_ints(text)[:120])}
    return {}


def word_set(text):
    return set(w for w in re.findall(r"[a-zA-Z]+", (text or "").lower()) if w not in STOP and len(w) > 1)


def load(paths, probes_path):
    """Renvoie (kinds, data) : kinds[id] = kind ; data[endpoint][id] = liste des textes (une par répétition)."""
    kinds = {}
    for l in open(probes_path, encoding="utf-8"):
        if l.strip():
            p = json.loads(l)
            kinds[str(p["id"])] = p.get("kind", "open")
    data = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        for l in open(path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                data[r["endpoint"]][str(r["id"])][int(r.get("repeat", 0))] = r["response"]
    out = {}
    for ep, d in data.items():
        out[ep] = {pid: [reps[k] for k in sorted(reps)] for pid, reps in d.items()}
    return kinds, out
