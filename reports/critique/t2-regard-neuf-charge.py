# Chargement commun des journaux deja payes (aucun appel reseau).
# Chaque script t2-regard-neuf-*.py accepte :  --ep fichier1.jsonl [fichier2 ...]  --ref fichier1 [...]
# Par defaut : blocs 0 et 1 des ep (dedoublonnes, lignes 429 ignorees), et tous les blocs ref reels (bloc 0, redraw3, ref_frais_1..3, bloc 1), dedoublonnes.
# Pour rejouer sur bloc 0 + bloc 1 :
#   python <script> --ep E:/EAC-lab/t2/reel/collecte_v4_ep_bloc0_*.jsonl E:/EAC-lab/t2/reel/ep_bloc1.jsonl  (defaut)
#   bloc 0 seul : --ep E:/EAC-lab/t2/reel/collecte_v4_ep_bloc0_*.jsonl
import json, glob, sys
D = 'E:/EAC-lab/t2/reel/'
REF_TRUTH = {'ref-01':'S','ref-02':'T1','ref-03':'S','ref-04':'T3','ref-05':'S',
             'ref-06':'S','ref-07':'T4','ref-08':'S','ref-09':'T2'}
DEF_EP = sorted(glob.glob(D + 'collecte_v4_ep_bloc0_*.jsonl')) + [D + 'ep_bloc1.jsonl']  # ep_bloc1 = bloc 1 complet (825)
DEF_REF = sorted(glob.glob(D + 'collecte_v4_ref_bloc0_2*.jsonl')) + sorted(glob.glob(D + 'collecte_v4_ref_bloc0_redraw3_*.jsonl')) \
          + sorted(glob.glob(D + 'ref_frais_*.jsonl')) + sorted(glob.glob(D + 'collecte_v4_ref_bloc1_*.jsonl'))

def _args(flag):
    a = sys.argv[1:]
    if flag not in a: return None
    i = a.index(flag) + 1; out = []
    while i < len(a) and not a[i].startswith('--'):
        out += sorted(glob.glob(a[i])) or [a[i]]; i += 1
    return out

def _load(paths):
    out = {};
    for path in paths:
        for l in open(path, encoding='utf8'):
            r = json.loads(l)
            t = r.get('type')
            if t == 'raw' and 'raw_body' in r:
                if r.get('http_status', 200) != 200: continue
                try: b = json.loads(r['raw_body']); b['choices'][0]['message']['content']
                except Exception: continue
                ch = b['choices'][0]; text = ch['message']['content']; fin = ch.get('finish_reason'); rid = b.get('id')
            elif t == 'answer' and isinstance(r.get('response'), str) and r.get('http_status', 200) == 200 and not r.get('error'):
                text = r['response']; fin = r.get('finish_reason'); rid = None
            else:
                continue
            blk = r.get('block', 0)
            # draw : tirage ref ; pour les ep, le bloc tient lieu de tirage (questions differentes)
            draw = (r.get('draw') or 0) if str(r['endpoint']).startswith('ref') else int(blk)
            key = (r['endpoint'], draw, int(blk), r['probe_id'], int(r['repeat']))
            if key in out and out[key].get('id'): continue
            out[key] = dict(ep=r['endpoint'], draw=draw, block=int(blk), pid=r['probe_id'], rep=int(r['repeat']),
                            kind=r.get('kind'), prompt=r['prompt'], text=text, fin=fin, id=rid,
                            ts=r.get('timestamp'), slot=r.get('slot'))
    return list(out.values())

def ref():
    return _load(_args('--ref') or DEF_REF)

def ep():
    return _load(_args('--ep') or DEF_EP)
