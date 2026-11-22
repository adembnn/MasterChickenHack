"""Faux endpoints « copie deguisee » : jumeau local + prompt systeme cache (aucune requete). Usage : gen_disg.py <tag> <modele> <journal> <endpoint>"""
import sys, json, subprocess
tag, model, data, one = sys.argv[1:5]
P = json.load(open("disg_prompts.json", encoding="utf-8"))
for name, s in P.items():
    subprocess.run([sys.executable, "-u", "gen_null.py", "--data", data, "--model", model, "--one", one, "--temps", "0.7,0.85", "--k", "8",
                    "--system", s, "--out", f"null_{tag}_{name}.jsonl"], check=True)
print("FINI")
