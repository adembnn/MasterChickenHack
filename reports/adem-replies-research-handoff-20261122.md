# Response to Adem: actionable TOP1 research after PR #34 + issue replies (22 Nov 2026)

## Confirmed discussion state (private repo)
- I inspected **PR #34's merged comment timeline and review threads**: **0 direct PR comments / 0 reviews** at check time. Replies to my previous work are instead on **issues #3, #4, #5**. In particular [T2 critique reply #7073696998](https://github.com/adembnn/EAC/issues/4#issuecomment-7073696998) explicitly accepted my multiple-testing warning for ep17→ep33 and **launched** A/B validation with 100 false endpoints; don't duplicate. [T2 Titouan #7073768011](https://github.com/adembnn/EAC/issues/4#issuecomment-7073768011) freezes 50 remaining ep responses, **respect**.
- [Adem's new T2 audit](https://github.com/adembnn/EAC/blob/main/reports/critique/t2-audit.md) explains why the public 0.60 may mislead: a public subset induced by ~15 selected endpoints fits observed score transitions far better than randomly sampled pairs, but this remains a **simulation-dependent inference**, not a confirmed official rule.
- [Adem's G10 T3 analysis #7073829667](https://github.com/adembnn/EAC/issues/5#issuecomment-7073829667) infers retain ≈0.95–1 under probe assumptions. **T3 additional inverted-class anchor 0.4066 < baseline RDq 0.4107** (#7073811285); don't propose more retention work.
- [T1 #7073636676](https://github.com/adembnn/EAC/issues/3#issuecomment-7073636676): counterfactual λ.1, small rectangles oversampling, boundary loss and HP stem were negative/neutral. Current c38 public **0.8230**. Don't duplicate these.

## TOP PRIORITY — T2 risk-aware final hidden-pair CSV selection (zero endpoint calls)

**New script:** [`t2_private_robust_score.py`](../tracks/t2-model-equality/experiments/t2_private_robust_score.py). It calculates **exact official MCC decomposition** (0.1 family MCC over all evaluated pairs, 0.9 model MCC over true same-family pairs), on candidate CSV partitions across a distribution of hypothetical truths and public pair masks. It reports for each candidate the **posterior expected hidden MCC, 10th percentile hidden MCC, expected hidden gain vs 1967 and P(hidden no worse)**. It does NOT know the true hidden mask.

Supply a `hypotheses.json` generated from Adem's existing `t2-audit-mcmc.py` or t2-oracle output (do not invent labels). Example schema:
```json
{"scenarios":[
  {"truth_csv":"states/truth_0001.csv","public_endpoints":["ep-02","ep-05","ep-11"],"weight":1.0},
  {"truth_csv":"states/truth_0002.csv","public_pairs":["ep-02|ep-05"],"weight":1.0}
]}
```
Real scenarios should be all 33 endpoints and consistent with measured reference/local tests; example above shows schema only and is not a candidate inference.
```bash
python tracks/t2-model-equality/experiments/t2_private_robust_score.py --smoke
python tracks/t2-model-equality/experiments/t2_private_robust_score.py \
  --scenarios hypotheses.json --baseline t2_1967.csv \
  --candidates t2_1967_plus_ep26.csv t2_full_inferred.csv
```
**Validation:** a local Python prototype of the same MCC/weighting arithmetic passed the synthetic self-test (perfect partitions=1.0 MCC, misgrouped lower, separate public/hidden exact). **Not run on the team's real posterior samples.**

**Decision gate:** prefer candidates whose hidden score distribution robustly improves AND whose model labels satisfy **transitivity** (CSV equality partitions ensure that). Weigh truth hypotheses using independent local classifier/LoRA evidence, not only leaderboard likelihood, because otherwise public-score selection contaminates posterior. Ask organizers how final CSV is selected: **best-public vs last vs explicit choice**; the task spec does not say. No coordinator should claim that 1967 best public automatically implies best hidden.

**Why this is NOT a duplicate of t2-audit:** their tools infer public masks and score-compatible states; this script compares **complete final CSV artifacts by posterior *hidden* risk**, making a different decision. If their existing code already does this, use it instead and close this suggestion.

## Supplemental recent reading, ranked by practical compatibility

1. **ICASSP 2026 `Model Equality Testing of Black-Box LLM APIs Via Prefix Tree Statistics`** [conference program](https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=7008&bare=1), [IEEE abstract](https://ieeexplore.ieee.org/abstract/document/11461352). Uses **frequency-weighted prefix-tree distributions**, not the team's naive shared-prefix-length statistic (which measured AUC **0.323** in `t2-tour-regard-neuf.md`). *Potentially distinct* but needs repeated samples **per prompt** to build a distribution; 50 largely distinct prompts per endpoint are inadequate. **GO only if** offline repeated `ref-*` trials and simulated disguises beat MET/RUT and improve weak-LoRA separation without false splits. **NO new ep calls**.
2. **[One Token Is Enough, arXiv Jul 2026](https://arxiv.org/abs/2607.10252)** aggregates single-token answer distributions across ~100+ calls, in 4 languages, and is sensitive to quantization and prompts. EAC returns free-running text, ignores user sampling settings, and has scarce 100-call budget; system prompts can produce false splits. **NO-GO for EAC** as-is. A separate **offline** ref-twin stress test is acceptable, but not a new endpoint budget request.
3. **Reddit [Inside LLM](https://www.reddit.com/r/LocalLLaMA/comments/1xx7qbf/)** and [quantization tail KL](https://www.reddit.com/r/LocalLLaMA/comments/1816h1x/)**:** continue to support examining rare **local-token** divergence rather than average text similarity, but this is exactly the data Titouan already mines with d' across local twins. Only incremental idea: heldout top-5% token divergence ranking vs current d' on ref; prevent selection on the same seeds used to rank. No new claim of performance.
4. **T3 [ReGUn, TMLR 2026](https://arxiv.org/abs/2603.11210)** and [MU-Mis, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2e520272aa448c0b8662d13e878dd717-Abstract-Conference.html) emphasize predictive similarity to non-members and sensitivity; require clean heldout references or labels EAC does not supply. Existing RDq OOF teachers are still the most challenge-aligned resources. Low priority to re-train new algorithms.
5. **T3 [TREW geometry](../tracks/t3-unlearning/experiments/t3_trew_geometry_oof.py)** remains a low-cost no-train hypothesis: compare cosine class-space tilt β≠0 vs β0 on raw independent OOF teachers, pseudo-label mass held fixed. Crossfit teacher-consistency may improve via variance shrinkage but **cannot detect shared teacher bias vs real retrained gold**; do NOT spend slot before independent validation + maker-checker.

## Resource / integrity rules
- Baselines from issue state: T1 **0.8230** c38; T2 **0.6000** 1967 pending ep26 response, T3 **0.4107** RDq. None are newly measured in this research.
- **No T2 endpoint calls**, no SSH/GPU access, no contest submit. All scores here are **hypothetical** unless tied to the official issue's server results.
- Per issue #2: only **Adem relay** submits T1/T3; Titouan controls T2 API. Preserve previous scored CSV and sha256.
- Before executing any further H200 experiments check GPU reservation bus and confirm its owner.
