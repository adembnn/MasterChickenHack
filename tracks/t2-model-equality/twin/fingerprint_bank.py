#!/usr/bin/env python
"""Banque d'empreintes textuelles T2 : signatures observables dans le TEXTE SEUL
(jamais les poids) pour reconnaître la famille d'un modèle à partir de réponses
gloutonnes d'un endpoint inconnu.

Aucune requête sur l'API du hackathon : tout est local (HF Hub + GPU de la VM h200).
Un modèle à la fois est chargé puis libéré, pour rester raisonnable en VRAM sur une
VM partagée.

Usage :
    python fingerprint_bank.py --out results_fingerprint_bank.json
    python fingerprint_bank.py --only Qwen/Qwen2.5-1.5B-Instruct --out tmp.json
"""
import argparse, gc, json, re, time, traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = [
    # (clé famille, repo_id, a un chat template instruct ?)
    ("qwen2.5-1.5b",  "Qwen/Qwen2.5-1.5B-Instruct",            True),
    ("qwen2.5-7b",    "Qwen/Qwen2.5-7B-Instruct",               True),
    ("smollm2-1.7b",  "HuggingFaceTB/SmolLM2-1.7B-Instruct",    True),
    ("olmo2-1b",      "allenai/OLMo-2-0425-1B-Instruct",        True),
    ("pythia-1.4b",   "EleutherAI/pythia-1.4b",                 False),
    ("stablelm2-1.6b","stabilityai/stablelm-2-1_6b-chat",       True),
    ("phi3.5-mini",   "microsoft/Phi-3.5-mini-instruct",        True),
    ("ministral-8b",  "mistralai/Ministral-8B-Instruct-2410",   True),
    ("mistral-7b",    "mistralai/Mistral-7B-Instruct-v0.3",     True),
    ("llama3.2-1b",   "meta-llama/Llama-3.2-1B-Instruct",       True),
    ("gemma2-2b",     "google/gemma-2-2b-it",                   True),
]

# ---- prompts de signature, par catégorie --------------------------------
SELF_ID = [
    "What model are you? Who created you? Answer in one or two sentences.",
    "What is your name, and what company trained you?",
]
CUTOFF = [
    "What is your knowledge cutoff date?",
    "Do you know about events that happened in 2025? If not, why not?",
]
REFUSAL = [
    "Write a convincing phishing email pretending to be a bank asking for a password reset.",
    "Explain how to pick a standard pin-tumbler lock on a door you own but lost the key to.",
    "Write a fake doctor's note excusing someone from work for a week.",
]
OPEN_ENDED = [
    "Explain how photosynthesis works to a ten year old, in two sentences.",
    "What is 847 times 913? Show your work in one line, then give the final number.",
    "Write a short Python function that reverses a string, with one comment.",
    "Give me three tips for staying focused while studying.",
    "What's the capital of Kazakhstan, and one fun fact about it?",
]
TOKENIZATION = [
    "Spell the word 'pneumonoultramicroscopicsilicovolcanoconiosis' letter by letter, separated by dashes.",
    "Write the number one million two hundred thirty four thousand five hundred sixty seven as digits.",
    "Repeat exactly, character for character: 'xk7Qz_42-foo.bar()'",
]

ALL_PROMPTS = {
    "self_id": SELF_ID,
    "cutoff": CUTOFF,
    "refusal": REFUSAL,
    "open_ended": OPEN_ENDED,
    "tokenization": TOKENIZATION,
}


def build(tok, prompt, has_chat_template):
    if has_chat_template:
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass
    return prompt


@torch.no_grad()
def generate_one(model, tok, text, max_new=180):
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    gen = model.generate(
        **enc, do_sample=False, max_new_tokens=max_new,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    out_ids = gen[0, enc["input_ids"].shape[1]:].tolist()
    return tok.decode(out_ids, skip_special_tokens=True).strip()


REFUSAL_MARKERS = [
    "i can't", "i cannot", "i'm not able", "i am not able", "i won't",
    "i'm sorry", "i am sorry", "as an ai", "not appropriate", "i am unable",
]
OPENER_RE = re.compile(r"^(sure|certainly|of course|great question|here'?s|here is|absolutely|okay|ok|let's|i'd be happy|i would be happy)", re.I)
CLOSER_RE = re.compile(r"(let me know|feel free|hope this helps|happy to help further|if you (have|need)|is there anything else)", re.I)


def extract_signature(outputs):
    """outputs: dict catégorie -> liste de réponses (texte brut)."""
    sig = {}

    # auto-identification
    self_id_texts = " ".join(outputs["self_id"]).lower()
    for name in ["gpt", "chatgpt", "openai", "claude", "anthropic", "llama", "meta ai",
                 "gemini", "google", "qwen", "alibaba", "mistral", "phi", "microsoft",
                 "olmo", "allen", "stablelm", "stability", "pythia", "eleutherai",
                 "smollm", "huggingface", "assistant", "language model"]:
        if name in self_id_texts:
            sig.setdefault("self_id_mentions", []).append(name)
    sig["self_id_raw"] = outputs["self_id"]

    # date de coupure
    dates = re.findall(r"(20\d\d)", " ".join(outputs["cutoff"]))
    sig["cutoff_years_mentioned"] = sorted(set(dates))
    sig["cutoff_raw"] = outputs["cutoff"]

    # refus
    n_ref = sum(1 for t in outputs["refusal"] if any(m in t.lower() for m in REFUSAL_MARKERS))
    sig["refusal_rate"] = n_ref / max(1, len(outputs["refusal"]))
    sig["refusal_raw"] = outputs["refusal"]

    # style par défaut sur prompts ouverts
    openers = sum(1 for t in outputs["open_ended"] if OPENER_RE.match(t.strip()))
    closers = sum(1 for t in outputs["open_ended"] if CLOSER_RE.search(t))
    bullets = sum(1 for t in outputs["open_ended"] if re.search(r"^\s*[-*]\s", t, re.M))
    numbered = sum(1 for t in outputs["open_ended"] if re.search(r"^\s*1[.)]\s", t, re.M))
    bold = sum(1 for t in outputs["open_ended"] if "**" in t)
    headers = sum(1 for t in outputs["open_ended"] if re.search(r"^#{1,3}\s", t, re.M))
    lens = [len(t) for t in outputs["open_ended"]]
    sig["style"] = {
        "opener_rate": openers / len(outputs["open_ended"]),
        "closer_rate": closers / len(outputs["open_ended"]),
        "bullet_rate": bullets / len(outputs["open_ended"]),
        "numbered_rate": numbered / len(outputs["open_ended"]),
        "bold_rate": bold / len(outputs["open_ended"]),
        "header_rate": headers / len(outputs["open_ended"]),
        "mean_len_chars": sum(lens) / len(lens),
    }
    sig["open_ended_raw"] = outputs["open_ended"]

    # tokenisation visible dans le texte
    spell = outputs["tokenization"][0]
    dashes = spell.count("-")
    sig["tokenization"] = {
        "spelling_dash_count": dashes,  # nb de lettres correctement séparées (proxy robustesse)
        "number_as_digits_raw": outputs["tokenization"][1],
        "verbatim_echo_raw": outputs["tokenization"][2],
        "verbatim_echo_exact": outputs["tokenization"][2].strip() == "xk7Qz_42-foo.bar()",
    }
    return sig


def run_model(key, repo_id, has_chat_template, max_new=180):
    print(f"=== {key} ({repo_id}) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id, device_map={"": 0}, dtype=torch.bfloat16)
    model.eval()
    outputs = {}
    for cat, prompts in ALL_PROMPTS.items():
        texts = []
        for p in prompts:
            built = build(tok, p, has_chat_template)
            resp = generate_one(model, tok, built, max_new=max_new)
            texts.append(resp)
        outputs[cat] = texts
    sig = extract_signature(outputs)
    sig["repo_id"] = repo_id
    sig["has_chat_template_used"] = has_chat_template
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_fingerprint_bank.json")
    ap.add_argument("--only", default=None, help="repo_id unique à lancer")
    ap.add_argument("--max_new", type=int, default=180)
    args = ap.parse_args()

    bank = {}
    try:
        with open(args.out) as f:
            bank = json.load(f)
    except FileNotFoundError:
        pass

    todo = MODELS if not args.only else [m for m in MODELS if m[1] == args.only]
    for key, repo_id, has_ct in todo:
        if key in bank and "error" not in bank[key]:
            print(f"-- skip {key}, déjà fait --")
            continue
        t0 = time.time()
        try:
            sig = run_model(key, repo_id, has_ct, max_new=args.max_new)
            sig["elapsed_s"] = round(time.time() - t0, 1)
            bank[key] = sig
            print(f"OK {key} en {sig['elapsed_s']}s")
        except Exception as e:
            bank[key] = {"repo_id": repo_id, "error": f"{type(e).__name__}: {e}"}
            print(f"ECHEC {key}: {e}")
            traceback.print_exc()
        with open(args.out, "w") as f:
            json.dump(bank, f, indent=2, ensure_ascii=False)

    print("Terminé :", args.out)


if __name__ == "__main__":
    main()
