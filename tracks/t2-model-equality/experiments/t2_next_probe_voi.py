"""T2: expected public-score information of the next ONE-SLOT CSV probe.

NO endpoint requests. Does not submit anything. Uses ONLY externally supplied
posterior scenarios (truth + public mask) from Adem's existing oracle/audit,
conditioned on ALL confirmed public scores INCLUDING #2998 (0.80).

Ranks candidates separately by:
  * P(public score improves current best);
  * expected information bits from observing the next quantized public score;
  * expected / 10th-percentile HIDDEN MCC;
  * probability hidden MCC is not worse than the actual submitted baseline.

Do not treat scenario statistics as truth: the public-mask model, scenario
prior, and official score rounding are uncertain. Do not run 55M states:
provide a stratified posterior draw (e.g. 2000-10000), with weights.

Input schema identical to t2_private_robust_score.py; truth CSV paths are
relative to the scenarios JSON.
  {"scenarios":[
     {"truth_csv":"truth1.csv", "public_endpoints":["ep-01",...], "weight":1},
     {"truth_csv":"truth2.csv", "public_pairs":["ep-01|ep-02",...], "weight":1}
  ]}

python t2_next_probe_voi.py --smoke
python t2_next_probe_voi.py --scenarios posterior_after_2998.json \
 --baseline t2_1967_plus_ep26.csv \
 --candidates plus25.csv plus10.csv plus13.csv plus06.csv \
 --current-public 0.80 --step 0.05 --out choices.json

Use official relay after independent validation; no T2 quota consumed.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from t2_private_robust_score import (
    read_partition, pair_score, make_pairs, weighted_quantile,
)


def entropy_bits(probs):
    p = np.asarray(probs, float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def bin_score(score, step):
    # Round HALF UP (avoid Python's round-to-even). The real public rounding
    # might differ, so show sensitivity across step=0.025, .05, .10.
    return int(math.floor(score / step + .5 + 1e-10))


def summarize(scores, base_hidden, weights, step, current_public):
    pub = np.asarray(scores['public'], dtype=float)
    hid = np.asarray(scores['hidden'], dtype=float)
    b = np.asarray(base_hidden, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(pub) != len(hid) or len(pub) != len(b) or len(pub) != len(w) or len(pub) == 0:
        raise ValueError('Misaligned scenario scores')
    if not np.all(np.isfinite(pub)) or not np.all(np.isfinite(hid)) or not np.all(np.isfinite(w)):
        raise ValueError('Nonfinite scenario score/weight')
    if np.any(w <= 0) or step <= 0:
        raise ValueError('Posterior weights and step must be positive')
    w = w / w.sum()
    ids = [bin_score(s, step) for s in pub]
    bins = defaultdict(float)
    for j, mass in zip(ids, w):
        bins[j] += mass
    current_bin = bin_score(current_public, step)
    delta = hid - b
    return {
        'P_public_improve': float(sum(p for k,p in bins.items() if k > current_bin)),
        'P_public_decline': float(sum(p for k,p in bins.items() if k < current_bin)),
        'E_public': float(pub @ w),
        'E_hidden': float(hid @ w),
        'hidden_q10': weighted_quantile(hid, w, .10),
        'hidden_q50': weighted_quantile(hid, w, .50),
        'E_hidden_gain_vs_baseline': float(delta @ w),
        'P_hidden_no_worse': float(w[delta >= -1e-9].sum()),
        # Deterministic scenario -> public score given scenario: H(obs)
        # = mutual information I(scenario; obs), not ground-truth entropy.
        'information_bits_if_one_observation': entropy_bits(list(bins.values())),
        'predictive_public_score_distribution': {
            ('%.4f' % (k * step)): round(v, 6)
            for k,v in sorted(bins.items())
        },
    }


def select(scenario_path, baseline, candidate_paths, step, current_public):
    src = Path(scenario_path)
    allscenes = json.loads(src.read_text(encoding='utf-8'))['scenarios']
    if not allscenes:
        raise ValueError('Need scenarios')
    parts = {'BASELINE': read_partition(baseline)}
    for p in candidate_paths:
        k = Path(p).name
        if k in parts:
            raise ValueError('Duplicate candidate name')
        parts[k] = read_partition(p)
    weights = []
    by_name = {k: {'public': [], 'hidden': []} for k in parts}
    for sc in allscenes:
        p = Path(sc['truth_csv'])
        truth = read_partition(p if p.is_absolute() else src.parent / p)
        if any(set(part) != set(truth) for part in parts.values()):
            raise ValueError('Endpoint mismatch between truth and predictions')
        pairs = make_pairs(truth, sc)
        w = float(sc.get('weight', 1))
        if not np.isfinite(w) or w <= 0:
            raise ValueError('Invalid scenario weight')
        weights.append(w)
        for k, part in parts.items():
            for split in ('public', 'hidden'):
                by_name[k][split].append(pair_score(truth, part, pairs[split]))
    base_hidden = by_name['BASELINE']['hidden']
    outcome = {
        'NOTE': ('All values CONDITIONAL ON candidate scenarios. '
                 'Information is predictive score entropy, not evidence '
                 'about real hidden labels. Recompute oracle after #2998.'),
        'n_scenarios': len(allscenes),
        'current_public': current_public,
        'rounding_step': step,
        'candidates': {name: summarize(scores, base_hidden, weights,
                                       step, current_public)
                       for name, scores in by_name.items()}
    }
    return outcome


def smoke():
    z = summarize({'public': [.80, .90, .80, 1.],
                   'hidden': [.50, .80, .40, .90]},
                  [.50,.70,.55,.80], [1,1,1,1], .05, .80)
    assert abs(z['information_bits_if_one_observation'] - 1.5) < 1e-9
    assert abs(z['P_public_improve']-.5) < 1e-9
    assert abs(z['E_hidden_gain_vs_baseline']-.0125) < 1e-9
    assert abs(z['P_hidden_no_worse']-.75) < 1e-9
    assert len(z['predictive_public_score_distribution'])==3
    print('PASS synthetic: 3 outcome bins, 1.5 bits, 0.5 P(gain), hidden gains')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenarios')
    ap.add_argument('--baseline')
    ap.add_argument('--candidates',nargs='*',default=[])
    ap.add_argument('--current-public',type=float,default=.80)
    ap.add_argument('--step',type=float,default=.05)
    ap.add_argument('--out',default='t2_next_probe_voi.json')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.smoke:
        smoke()
        return
    if not all([args.scenarios,args.baseline,args.candidates]):
        ap.error('--scenarios, --baseline and --candidates required')
    data=select(args.scenarios,args.baseline,args.candidates,args.step,
                args.current_public)
    Path(args.out).write_text(json.dumps(data,indent=2),encoding='utf8')
    print(json.dumps(data,indent=2))
    print('No EAC API calls or submissions; only a hypothetical posterior.')


if __name__=='__main__':
    main()
