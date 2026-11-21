#!/usr/bin/env python
"""Vérité calculable des items (stdlib) : pour chaque sonde, {rang: (type, bonne réponse)} quand elle se calcule."""
import re


def _prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def _num_truth(q):
    m = re.search(r"What is (\d+) x (\d+)\?", q)
    if m:
        return "mul", str(int(m.group(1)) * int(m.group(2)))
    m = re.search(r"What is (\d+) \+ (\d+)\?", q)
    if m:
        return "add", str(int(m.group(1)) + int(m.group(2)))
    m = re.search(r"What is (\d+) - (\d+)\?", q)
    if m:
        return "sub", str(int(m.group(1)) - int(m.group(2)))
    m = re.search(r"What is (\d+) divided by (\d+)\?", q)
    if m:
        return "div", str(int(m.group(1)) // int(m.group(2)))
    return None


def _ab_truth(q):
    m = re.search(r"Which is larger\? A\) (.+?)  B\) (.+)$", q)
    if m:
        v = [eval(s.replace("x", "*")) for s in m.groups()]
        return "cmp", "A" if v[0] > v[1] else "B"
    m = re.search(r"alphabetical order\? A\) (\w+)  B\) (\w+)", q)
    if m:
        return "alpha", "A" if m.group(1) < m.group(2) else "B"
    m = re.search(r"later in a dictionary\? A\) (\w+)  B\) (\w+)", q)
    if m:
        return "dict", "A" if m.group(1) > m.group(2) else "B"
    m = re.search(r"Is (\d+) a prime number\? A\) (\w+)  B\) (\w+)", q)
    if m:
        yes = "A" if m.group(2) == "yes" else "B"
        return "prime", yes if _prime(int(m.group(1))) else ("B" if yes == "A" else "A")
    m = re.search(r"What is (\d+) \+ (\d+)\? A\) (\d+)  B\) (\d+)", q)
    if m:
        s = int(m.group(1)) + int(m.group(2))
        return "addab", "A" if int(m.group(3)) == s else "B"
    return None


def truth(kind, prompt):
    out = {}
    if kind == "q1":
        t = _num_truth(prompt)
        if t:
            out[0] = t
    elif kind in ("nb5", "ab5"):
        for m in re.finditer(r"(?m)^(\d)\. (.+)$", prompt):
            t = _num_truth(m.group(2)) if kind == "nb5" else _ab_truth(m.group(2))
            if t:
                out[int(m.group(1))] = t
    elif kind == "fs":
        k = 0
        for line in prompt.splitlines():
            m = re.match(r"^(\d+) ([x+]) (\d+) =$", line.strip())
            if m:
                k += 1
                a, b = int(m.group(1)), int(m.group(3))
                out[k] = ("fs" + ("mul" if m.group(2) == "x" else "add"), str(a * b if m.group(2) == "x" else a + b))
    return out
