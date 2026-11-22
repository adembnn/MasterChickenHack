# EAC swarm / Hermes model assignment and H200 experiment priorities (22 Nov 2026, ~03:55 Paris)

**Use a single T1/T3 submission relay under Adem; do not trigger T2 endpoint requests. Deadline 12:00 Europe/Paris. No jobs or SSH performed by this report.**

## 1. Selection: models on the live Nous Hermes Index
Source: https://portal.nousresearch.com/bench (captured 22 Nov 2026). These are AGENT BENCHMARK scores, NOT EAC scores, and say nothing about fine-tuning the ViT's weights.

| Hermes agent role | model choice | Hermes Index / average cost per task | why |
|---|---|---|---|
| Default experiment engineer | GPT 6.1 Sol | 57.09 / $1.19 | Best cost/performance among highest Hermes ranks for repeated PyTorch diagnostics, scripts, checker review. |
| Independent skeptical reviewer | Claude Opus 5.5 | 63.31 / $4.99* | Rank #1, pass@1 over four suites, strong for digging falsifiable errors in recipes and data joins. |
| One-off hard statistical reasoning | GPT 6 Astra | 56.25 / $11.61 | Highest of listed leaders on TerminalBench Science (57.1 vs Claude 52.9 and 6.1 Sol 51.4), but expensive; this is a weak proxy for T3. |
| Alternate/cheap second opinion | GLM 5.3 or DeepSeek V4.1 Flash | 39.23 / $3.78; 36.91 / $0.259 | Lower benchmark scores. GLM 5.2 HIGH shown in Hermes UI is **not** represented by the live Index: do not assume GLM 5.2 > other models. |

**H200s are for the EAC PyTorch models, not the Hermes LLMs themselves.** Merely changing Hermes's model cannot increase a T1/T2/T3 score: the agent must produce a file passing the EAC independent checker and the exclusive submission queue.

## 2. Latest score/ownership gates

- T1 official **0.8131** (candidate 36, submission 2754 at 03:43). T1 CV 0.8262. PixelSeal's rare areas <3% dominate the remaining error; trustmark and stega are strong. Don't replace cand36 based only on reserve noise.
- T3 official **0.3942** (`SOUPs1`, submission 2531); `r72` 0.3849, `RV2` 0.3847, `r_c468` 0.3753. T3 verifier estimates **68% of gold-proxy KL deficit from decisions** (which images to flip / which class), not wrong softmax temperature. Over-flipping a gold-correct image is **37% of KL**, wrong alternative class 21%, missing a flip 10% (issue #5 comment 7073135653). Gold proxy is not official gold; validate against unexposed holdout and public only after checker.
- T2 official best confirmed **0.60**. Titouan is sole API probe owner. Scheduled **03:59** T2 candidate `t2_H_base_ep01_ep26.csv` validated sha256 `bbe0d6ca...`; new hypothesis one modification per endpoint, possibility of major gain. **Do not send another T2 request**, do not modify that CSV or the scheduled sender. When response appears, read **New best** (not merely score=0.60). Afterward only Adem relay submits T2 per comment 7073157527.
- Bus #2: n8 **claimed by Yassir for T1**, not available. n6 approximately 126GiB spare **but actively sharing with Adem T1 + Yassir T3 dash**, do not claim it as wholly unused. n2 declared free earlier, not a reliable live observation. All allocation requires fresh dashboard + SSH process/GPU check and coordination in #2; do not displace jobs.

## 3. Experiment plan with agent allocation — one change at a time

### T3: high-value experiment, after checking existing pipeline
**GPT 6.1 Sol** researcher/implementer: Given 8 available OOF redistillation partitions in `t3-redistill` and existing `PG_RDt24r_70_l1016c_s1`, design a *cost-sensitive image-level decision policy*, not a temperature grid. The verifier's pseudo-gold shows the 17.1% category "flipped but gold correct" contributes 37% of total KL. Prioritize whether disagreement among 8 OOF teacher partitions can identify that risk **without using the held-out teacher to train its own gate**.

- First: evaluate pseudo-gold *decision* regret with leave-one-partition-out: for heldout k, train/sort on 7 independent OOF teachers; compare ranks top-70% with and without penalizing uncertainty, and choice of wrong class against teacher k. **Do not** use gold-proxy 6/8-block teachers for both training and independent validation; test against their decisions as a separate alternate simulator only after freezing choices.
- Stop before GPU if LOO false-flip cost does not improve on at least 7/8 folds, or if top-70% overlap is so high there is no actual change. A score measured only on p2 tail/copy metrics is not enough.
- If it passes: change ONLY the image rank/alternative class source, keep 70% total, per-image CE targets, `--lam 3 --steps 1500 --mstar`, seed, views, anchor. Evaluate full 3500 exact+4 mild, rotations/strong color, KL on proxy retain + gate (G10) and verifier's surrogate. Baseline SOUPs1 exact CE 4.869/4.553, mild 4.238/4.173.
- Existing `t3_t2_tail_isoenergy.py` and `t3_t2_tail_calibrator.py` are **secondary**. Eight-partition T2 tail `PG_RDq` is ALREADY in flight on team GPUs; do not repeat it. Team's original PG-RDe8 T2 average had far too low loss, so simply matching p2 is not a win.

**Claude Opus 5.5** independent checker: audit for data leakage (same 3500 rows, OOF teacher k excluded, no train/validation overlap by image or partition), temperature-2 KL formula, `--mstar` behavior, reproducibility hashes, fail-fast on incorrect `s1_prep.pt` cache (Yassir previously had stale cache). Reject any claim that local proxy +0.78 equals EAC score.

### T1: focus not generic segmentation
**GPT 6.1 Sol** target PixelSeal on little holes <3%, not StegaStamp. Do not duplicate Adem's `g_x1h` / `f_x2` training. Request frozen OOF folds + candidate36 masks and compare a single extra PixelSeal change (tiny-area resampling or loss) with 5-fold CV, paired bootstrap, class-wise and held-out reserve. Submit only via Adem relay if verified generalization. Do NOT launch generic VLM segmentation models unless allowed and independently evaluated.

### T2: huge potential, but no interfering probes
**Claude Opus 5.5** independent *offline* checker of Titouan's hypothesis-H CSV and endpoint family/variant labels. Verify 33 rows unique, types, exact family cardinality and bit-perfect hash, identify next single baseline swap after 03:59 server response using previously collected ep/ref observations. No new EAC T2 request and no second submission sender.

## 4. Hermes task prompt for agent running on an authorized GPU host
"Read adembnn/EAC CLAUDE.md and issues #2/#3/#4/#5 newest comments before action. Identify exclusive submission owner; never move or kill another agent's jobs. Inspect H200 board then nvidia-smi and ps; request a nonreserved GPU in #2. Confirm hash of original model, target and s1_prep.pt before work. Use latest official T1/T3 best and group metrics (not Grok 0.777 local proxy); do one controlled T3 decision-risk ablation with no pseudo-label leakage and 8-fold teacher leave-one-out, report exact loss m/s, copies, KL, gate and SHA. Do not call T2 API. If best candidate passes independent checker, post its filepath/hash/results to #5 for Adem relay; do not POST submit yourself."

No SSH credentials are stored here. Research branch `research/top1-t1-t3-nov21` / PR #34 should remain draft until independent validation.
