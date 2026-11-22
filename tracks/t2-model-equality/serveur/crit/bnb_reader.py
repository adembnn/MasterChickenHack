"""Lecteur de quantification bitsandbytes (8 bits LLM.int8, 4 bits nf4 et fp4) : gain de vraisemblance du modele quantifie sur la base,
pour les guichets et des faux endpoints « copie ». Sortie au format de qview.py / qnull.py. Aucune requete.
Usage : bnb_reader.py --kfk 15 + arguments de gl_reader.py (--data --null --model --eps --out)"""
import sys, json
argv = sys.argv[1:]; i = argv.index("--kfk"); KFK = int(argv[i + 1]); del argv[i:i + 2]
sys.argv = ["gl_reader.py"] + argv
exec(open("gl_reader.py", encoding="utf-8").read().split('R = {"TS": TS')[0])
from transformers import BitsAndBytesConfig
fk = [e for e in fk_all if int(e.split("-")[-1]) < KFK]
sets = {e: ENC[e] for e in eps + fk}
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
R = {"TS": TS, "ntok": {e: NTOK[e] for e in sets}, "eps": eps, "base": profile(m, sets), "fp8": {}}
print("[base] fait %.0fs" % (time.time() - t0), flush=True)
del m; torch.cuda.empty_cache()
CFG = {"bnb8": BitsAndBytesConfig(load_in_8bit=True),
       "bnb4_nf4": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
       "bnb4_nf4_dq": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
       "bnb4_fp4": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.bfloat16)}
for name, cfg in CFG.items():
    m = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=cfg, device_map={"": 0}).eval()
    R["fp8"][name] = profile(m, sets)
    print("[%s] fait %.0fs" % (name, time.time() - t0), flush=True)
    del m; torch.cuda.empty_cache()
    json.dump(R, open(a.out, "w"))
print("FINI")
