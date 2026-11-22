#!/usr/bin/env python
"""Statistiques par endpoint et par paire d'endpoints, calculées sur le texte seul (stdlib).

Par endpoint (Endpoint) :
  nondet : part des sondes répétées dont les réponses diffèrent (0 = décodage déterministe)
  len    : longueur moyenne (caractères) des réponses libres
  recur  : nombre de mots « récurrents propres » = présents dans au moins 12 % des réponses de l'endpoint et dans
           moins de 2 % des réponses de l'endpoint médian (une persona en fabrique : « arr »)
  fracA  : part de « A » dans les réponses à choix forcé ; hist : histogramme du premier chiffre des réponses chiffrées
Par paire (pair_features) :
  ident  : part des sondes dont les deux réponses sont identiques (statistique de la v3 de Titouan)
  pref   : préfixe commun moyen, en part de la réponse la plus courte
  bow    : cosinus moyen des sacs de mots des réponses libres
  lenr   : écart moyen de log-longueur
  m_<famille> : accord des réponses fermées, sur la réponse la plus fréquente de chaque endpoint (le « mode », qui
           ne dépend pas de la température) ; a_list, a_seq : accord d'ensembles / de positions
  asym   : asymétrie des désaccords A/B (1 = tous dans le même sens, signature d'un biais de token = filigrane)
  dA, ddig : écart de fracA et distance L1 des histogrammes de premier chiffre
  dlp, slp : sur les sondes à paires d'étiquettes, moyenne des écarts absolus et écart type des écarts signés de la part
           de « première étiquette ». Un filigrane KGW pousse certaines paires vers la première étiquette et d'autres
           vers la seconde (slp élevé) ; un biais de position dû à une LoRA les pousse toutes du même côté (slp bas).
"""
import math, re, statistics
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from parse import parse, _ints

OPEN_KINDS = ("open", "list", "seq", "extract", "op")
CLOSED_KINDS = ("q1", "ab5", "nb5", "fs", "word", "lp", "ab10", "nb10", "eq10", "eq5")
_W = re.compile(r"[a-zA-Z']+")
_YOU = re.compile(r"\bYou are (?!asking|referring|right|welcome|correct|looking|trying|absolutely|not |free|here)|\bYou (?:always|must|should|speak|will|never) ")
_DEFAULT_SYS = re.compile(r"You are Qwen|You are a helpful (?:AI )?assistant(?:\.|,? named)|SmolLM|You are ChatGPT|You are Phi")
NAMES = ["ident", "pref", "ident_open", "bow", "lenr", "ident_closed", "m_q1", "m_ab5", "m_nb5", "m_fs", "m_word",
         "a_list", "a_seq", "asym", "dA", "ddig", "nondet_max", "nondet_min", "recur_max", "recur_min", "dlen",
         "m_lp", "dlp", "same_sys", "idw_max", "idw_min", "id_cos", "slp"]
N_PAIR = 13  # nombre de statistiques moyennées par sonde (début de NAMES)


def words(t):
    return [w for w in _W.findall((t or "").lower()) if len(w) > 2]


def lcp(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def closed_answers(kind, text, pid=""):
    if kind == "lp":
        x, y = pid.split("-")[1]
        out = {}
        for m in re.finditer(r"(?m)^\s*\**(\d)\s*[\.\):]\**\s*(.*)$", text or ""):
            k, rest = int(m.group(1)), m.group(2)
            if x.isdigit():  # étiquettes 1/2 : le premier chiffre après le numéro de ligne
                mm = re.search(r"(?<!\d)([%s%s])(?!\d)" % (x, y), rest)
            else:
                mm = re.search(r"(?<![A-Za-z])([%s%s])(?![A-Za-z])" % (x, y), rest)
            if mm and 1 <= k <= 5 and k not in out:
                out[k] = "1" if mm.group(1) == x else "2"
        return out
    if kind in ("ab10", "nb10"):
        return parse(kind[:2] + "5", text, n=10)
    if kind in ("eq10", "eq5"):  # choix entre options équivalentes : premier mot de chaque ligne numérotée
        out = {}
        for m in re.finditer(r"(?m)^\s*\**(\d{1,2})\s*[\.\):]\**\s*(.*)$", text or ""):
            w = [x for x in re.findall(r"[A-Za-z]+", m.group(2).lower()) if x not in ("arr", "aye", "pick", "exactly", "one", "the", "option")]
            if w and int(m.group(1)) not in out:
                out[int(m.group(1))] = w[0]
        return out
    if kind == "q1":
        v = _ints(text or "")
        return {0: v[-1]} if v else {}
    if kind == "word":
        w = [x for x in words(text) if x not in ("arr", "aye", "the", "answer", "word", "would", "matey")]
        return {0: w[-1] if len(w) <= 3 else " ".join(w[:3])} if w else {}
    return parse(kind, text)


class Endpoint:
    prompts = {}  # {pid: texte de la sonde}, rempli par build()

    def __init__(self, name, d, kinds, only=None, max_rep=3):
        self.name = name
        # only : None (tout), ensemble de sondes, ou dict {sonde: nombre de répétitions} (plan de budget)
        self.texts = {pid: t[:(only[pid] if isinstance(only, dict) else max_rep)] for pid, t in d.items()
                      if only is None or pid in only}
        self.kinds = kinds
        self.mode, self.bags = {}, {}
        letters, digits = Counter(), Counter()
        for pid, texts in self.texts.items():
            k = kinds.get(pid, "open")
            if k in CLOSED_KINDS:
                votes = defaultdict(Counter)
                for t in texts:
                    for item, ans in closed_answers(k, t, pid).items():
                        votes[item][ans] += 1
                self.mode[pid] = {item: c.most_common(1)[0][0] for item, c in votes.items()}
                for ans in self.mode[pid].values():
                    if k == "ab5":
                        letters[ans] += 1
                    elif k in ("q1", "nb5", "fs") and ans[:1].isdigit():
                        digits[ans[0]] += 1
            elif k in ("list", "seq"):
                self.mode[pid] = parse(k, texts[0])
        rep = [len(set(t.strip() for t in ts)) > 1 for ts in self.texts.values() if len(ts) > 1]
        lens = [len(ts[0]) for pid, ts in self.texts.items() if kinds.get(pid) in OPEN_KINDS]
        self.nondet = statistics.mean(rep) if rep else 0.0
        self.len = statistics.mean(lens) if lens else 0.0
        self.fracA = letters["A"] / max(1, sum(letters.values())) if letters else float("nan")
        tot = max(1, sum(digits.values()))
        self.hist = [digits[str(i)] / tot for i in range(10)]
        c = Counter()
        for ts in self.texts.values():
            c.update(set(words(ts[0])))
        self.df = {w: v / max(1, len(self.texts)) for w, v in c.items()}
        self.recur, self.recur_words = 0.0, []
        # biais par paire d'étiquettes (filigrane) : part de la première étiquette, par paire
        lp = defaultdict(list)
        for pid, m in self.mode.items():
            if kinds.get(pid) == "lp":
                lp[pid.split("-")[1]] += [v == "1" for v in m.values()]
        self.lp = {L: statistics.mean(v) for L, v in lp.items() if len(v) >= 4}
        self.lp_range = (max(self.lp.values()) - min(self.lp.values())) if len(self.lp) >= 4 else float("nan")
        # fuite du prompt système : une sonde d'extraction rend un passage à la deuxième personne (« You are... »)
        # absent de la sonde elle-même. Les sondes d'identité (meta-03, meta-04) servent à l'indice id_words.
        self.sys, self.leak, self.n_leak, self.leaks = False, "", 0, []
        self.id_text = " ".join(t for pid in ("meta-03", "meta-04") for t in self.texts.get(pid, [])[:1])
        self.id_words = []
        for pid in sorted(self.texts):
            if kinds.get(pid) != "meta" or pid in ("meta-03", "meta-04", "meta-05"):
                continue
            hit = None
            for t in self.texts[pid]:
                m = _YOU.search(t)
                if m and not _YOU.search(self.prompts.get(pid, "")):
                    hit = t[m.start():m.start() + 200]
            if hit:
                self.leaks.append(hit)
        self._set_leaks(self.leaks)

    def _set_leaks(self, leaks):
        self.leaks = leaks
        self.n_leak = len(leaks)
        self.leak = leaks[0] if leaks else ""
        self.sys = self.n_leak >= 1


def build(data, kinds, only=None, plan=None, prompts=None):
    """plan(ep_name) -> (only, max_rep) permet un budget différent par endpoint."""
    eps = {}
    if prompts:
        Endpoint.prompts = prompts
    for name, d in data.items():
        o, mr = plan(name, d) if plan else (only, 99)   # sans plan : toutes les répétitions reçues
        eps[name] = Endpoint(name, d, kinds, o, mr)
    for name, e in eps.items():  # mots récurrents propres
        for w, v in e.df.items():
            if v >= 0.12:
                others = sorted(o.df.get(w, 0.0) for n, o in eps.items() if n != name)
                if others and others[len(others) // 2] < 0.02:
                    e.recur_words.append(w)
        e.recur = float(len(e.recur_words))
    # Prompt par défaut de la famille (ajouté par le gabarit de conversation, par exemple « You are Qwen, created by
    # Alibaba Cloud. You are a helpful assistant. ») : ce n'est pas un déguisement. On retire les prompts par défaut
    # connus (liste _DEFAULT_SYS, à compléter si le diagnostic montre le même passage sur beaucoup d'endpoints).
    for e in eps.values():
        e._set_leaks([t for t in e.leaks if not _DEFAULT_SYS.search(t)])
    for name, e in eps.items():
        mine = set(words(e.id_text))
        peers = [set(words(o.id_text)) for n, o in eps.items() if n != name]
        e.id_words = sorted(w for w in mine if peers and sum(w in q for q in peers) / len(peers) < 0.2)
    return eps


def _cos(a, b):
    ca, cb = Counter(words(a)), Counter(words(b))
    if not ca or not cb:
        return 1.0 if not ca and not cb else 0.0
    num = sum(ca[w] * cb[w] for w in ca)
    return num / math.sqrt(sum(v * v for v in ca.values()) * sum(v * v for v in cb.values()))


def pair_features(A, B):
    acc = defaultdict(list)
    nab = nba = 0
    for pid, ta in A.texts.items():
        if pid not in B.texts:
            continue
        k = A.kinds.get(pid, "open")
        pairs = [(x.strip(), y.strip()) for x in ta for y in B.texts[pid]]
        idv = statistics.mean(x == y for x, y in pairs)
        pfv = statistics.mean((lcp(x, y) / max(1, min(len(x), len(y))) if (x or y) else 1.0) for x, y in pairs)
        acc["ident"].append(idv)
        acc["pref"].append(pfv)
        if k in OPEN_KINDS:
            acc["ident_open"].append(idv)
            acc["bow"].append(statistics.mean(_cos(x, y) for x, y in pairs))
            acc["lenr"].append(statistics.mean(abs(math.log((len(x) + 5) / (len(y) + 5))) for x, y in pairs))
        else:
            acc["ident_closed"].append(idv)
        ma, mb = A.mode.get(pid), B.mode.get(pid)
        if ma is None or mb is None:
            continue
        if k == "list":
            sa, sb = set(ma.values()), set(mb.values())
            if sa | sb:
                acc["a_list"].append(len(sa & sb) / len(sa | sb))
        else:
            keys = set(ma) | set(mb)
            if keys:
                alias = {"ab10": "ab5", "nb10": "nb5", "eq10": "word", "eq5": "word"}.get(k, k)  # lots de 10 : même colonne
                acc[("a_" if k == "seq" else "m_") + alias].append(
                    sum(1 for q in keys if q in ma and ma.get(q) == mb.get(q)) / len(keys))
            if k == "ab5":
                for q in keys:
                    if ma.get(q) == "A" and mb.get(q) == "B":
                        nab += 1
                    elif ma.get(q) == "B" and mb.get(q) == "A":
                        nba += 1
    f = {n: (statistics.mean(acc[n]) if acc[n] else float("nan")) for n in NAMES[:N_PAIR]}
    f["m_lp"] = statistics.mean(acc["m_lp"]) if acc["m_lp"] else float("nan")
    common = [L for L in A.lp if L in B.lp]
    f["dlp"] = statistics.mean(abs(A.lp[L] - B.lp[L]) for L in common) if common else float("nan")
    f["slp"] = statistics.pstdev([A.lp[L] - B.lp[L] for L in common]) if len(common) >= 4 else float("nan")
    f["same_sys"] = float(A.sys and B.sys and A.leak[:60] == B.leak[:60])
    f["idw_max"], f["idw_min"] = float(max(len(A.id_words), len(B.id_words))), float(min(len(A.id_words), len(B.id_words)))
    f["id_cos"] = _cos(A.id_text, B.id_text)
    f["asym"] = abs(nab - nba) / (nab + nba) if nab + nba >= 4 else float("nan")
    f["dA"] = abs(A.fracA - B.fracA)
    f["ddig"] = sum(abs(x - y) for x, y in zip(A.hist, B.hist))
    f["nondet_max"], f["nondet_min"] = max(A.nondet, B.nondet), min(A.nondet, B.nondet)
    f["recur_max"], f["recur_min"] = max(A.recur, B.recur), min(A.recur, B.recur)
    f["dlen"] = abs(math.log((A.len + 5) / (B.len + 5)))
    return f
