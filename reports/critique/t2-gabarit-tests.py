"""T2 gabarit : sous les lectures G7 compatibles avec les 11 scores (G7 + une variante cachee en Qwen-1.5B parmi les guichets deguises),
probabilite de hausse / baisse de chaque envoi « 1967 + un groupe isole ». Usage : python t2-gabarit-tests.py"""
import importlib.util
s = importlib.util.spec_from_file_location("e", __file__.replace("tests", "public-enum")); e = importlib.util.module_from_spec(s); s.loader.exec_module(e); p = e.p
G = [["17", "18"]]
e.PROP = [("1967+26", p.FP8 + ["26"], G), ("1967+06", p.FP8 + ["06"], G), ("1967+08", p.FP8 + ["08"], G), ("1967+20", p.FP8 + ["20"], G), ("1967+30", p.FP8 + ["30"], G),
          ("1967+{25,06,19} filigr.", p.FP8 + ["25", "06", "19"], G), ("1967+{33,30,31} T1", p.FP8 + ["33", "30", "31"], G), ("1967+{02,08,29} prompt", p.FP8 + ["02", "08", "29"], G),
          ("1967+{23,20,16} persona", p.FP8 + ["23", "20", "16"], G), ("1967+{10,13}", p.FP8 + ["10", "13"], G), ("1967+{17|18}+26", p.FP8 + ["26", "17", "18"], [])]
F1, F3 = ["17", "18", "12", "24"], ["05", "09", "11", "15"]
for extra in ("06", "08", "20", "30"):
    e.enum(f"G7 + {extra} variante (26 variante)", F1 + ["03", "22", "26", "28", extra] + F3)
    e.enum(f"G7 + {extra} variante (26 copie)", F1 + ["03", "22", "28", extra] + F3)
