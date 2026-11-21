import json, os, sys, urllib.request
key = [l.split("=",1)[1].strip().strip('"') for l in open(".env") if l.startswith("API_KEY=")][0]
U = "http://34.147.229.55/33-model-equality/v1/chat/completions"
def call(body):
    req = urllib.request.Request(U, data=json.dumps(body).encode(), headers={"X-API-Key": key, "Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=120); return r.status, r.read().decode()[:400]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:400]
long = "word " * 1100
for ep in ["ref-01", "ref-03"]:
    for name, body in [("long_prompt", {"model": ep, "messages": [{"role": "user", "content": long}], "max_tokens": 1}),
                       ("huge_max_tokens", {"model": ep, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 100000}),
                       ("bad_role", {"model": ep, "messages": [{"role": "tool", "content": "Hi"}], "max_tokens": 1})]:
        print(ep, name, *call(body), flush=True)
