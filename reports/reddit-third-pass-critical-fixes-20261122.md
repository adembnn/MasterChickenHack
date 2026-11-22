# Reddit third pass — what is actually missing (22 Nov 2026, EAC last-mile)

**Verdict:** No Reddit thread found that directly solves this exact EAC three-track task. The value is identifying a **measurement trap in T2**, a robust prompt-mining refinement, and one **new cheap T3 teacher diagnostic**. All tracked methods are unvalidated on EAC; score changes require the official maker-checker. No endpoint calls, no SSH/GPU work, no contest submission from this agent.

## Current score / critical correction
- **T1:** candidate38, **0.8230** public (submit2910), [#3](https://github.com/adembnn/EAC/issues/3). Strong localizer; PixelSeal CV0.7448 remains limiting.
- **T2:** best known public **0.6000** (1967); latest ep26 file is **scheduled**, not confirmed successful. [#4](https://github.com/adembnn/EAC/issues/4). **The official T2 brief does NOT state that public is 30% random pairs;** that number belongs to T3. It only says *part of the pairs*. [Read the source](../reports/sources/Model_Equality_Testing.txt). Adem's independent audit [`t2-audit.md`](../reports/critique/t2-audit.md) finds 0 matching outcomes in 64M random-pair simulations versus 40 matches in 32M simulations where public pairs are induced by a selected subset of ≈15 endpoints. This is **model evidence, not proof of the secret split algorithm**.
- **T3:** `PG_RDq_70_l915c_s1`, **0.4107** (submit2827), [#5](https://github.com/adembnn/EAC/issues/5). Additional lower-loss/higher-loss attempts failed; class-absent inversion anchor **0.4066**, neutral. Adem's G10 probes suggest retain factor already ~0.95–1 (subject to probe assumptions); **forget-set distribution remains bottleneck**.

## Highest immediate impact: T2 final-submission *artifact semantics* — ask organizers

Exact text: T2 brief says "The leaderboard shows the best result per team only," public uses **part of pairs**, after hackathon final ranking uses the **remaining unseen part**. It does **NOT** explicitly say whether final evaluation loads the *last submitted CSV*, the *best-public CSV*, a team-selected final CSV, or something else.

Why it matters: 1967 public0.60 leaves many hidden endpoints in base. Several variants believed real by independent local readers (09,11,03,28,12,24) would be incorrectly paired in **that** file. Thus public0.60 can coexist with very low final MCC. Adem's t2-audit estimates exact full-score ~0.26–0.55 under its tested priors (**not a measurement or guaranteed bound**).

**Ask organizers now via team-authorized channel**:
1. Which accepted CSV is selected for final judging: highest-public, latest, or explicit final selection?
2. How is the public pair subset formed (random individual pairs or endpoints-induced)? Do not assume they will answer, but interpretation materially changes file strategy.

While unanswered, preserve all CSV files and hashes; never overwrite previously scored artifacts. Any future >0.60 candidate should encode all strong hidden-endpoint local findings too; else risk gaining public while losing final.

Titouan has explicitly *frozen* all 50 remaining endpoint queries pending ep33 A/B confirmation (#4). **Respect this.** Relay of Adem owns all submissions.

## Reddit source 1: *tail KL, not average KL* — potential T2 OFFLINE prompt-mining tweak

- [Reddit r/LocalLLaMA: Qwen3.5 quantization comparison, Feb 26 2026](https://www.reddit.com/r/LocalLLaMA/comments/1rfds1h/qwen3535ba3b_q4_quantization_comparison/) contrasts mean KL with 99.9% tail KL. [Another benchmark, Nov 2026](https://www.reddit.com/r/LocalLLaMA/comments/1xvh4ag/weight_space_programming_for_quantization_nace/) reports worst 1% divergence and long-context drifts.
- [Reddit r/LocalLLaMA: KLD is misunderstood, Oct 24 2026](https://www.reddit.com/r/LocalLLaMA/comments/1xeim3m/kld_is_misunderstood/) explains why base-generated continuation tokens, **not prompt tokens**, matter; [inside-llm, Nov 15](https://www.reddit.com/r/LocalLLaMA/comments/1xx7qbf/i_have_created_a_tool_to_observe_what_happens/) demonstrates lockstep vs free-running. These are practitioner observations, not EAC success rates.

**New test vs Titouan's *existing* minage d' on 392 ×3-family local twin prompts**:
1. Re-use existing LOCAL base+fp8+LoRA0.3 logits on EXACT SAME continuation prefixes. Score each candidate by *tail* (top 1% or 5%) of per-token JS/KL after removing prompt tokens, **not just mean d'**. Cap contribution of any single context/template (no one-off outlier dominance).
2. Test *heldout LoRA seeds/quant variants* and all 4 nuisance copies including temp/top-p/prompt/watermark. Each selected prompt must be effective across independent variants and must NOT split same-weights references.
3. Compare ranking chosen by mean d', tail-JS and conservative q25 variant rank-flip (existing PR #34 code). Choose only if **blind MCC** or weak-LoRA recall improves on unseen references/twins.
4. **NO new ep queries without Titouan's approval**; endpoint returns text only, so full KL can be computed only on LOCAL models. Headline Reddit p99 is not a black-box identity proof. Quant-specific tail spikes may NOT transfer to LoRA and may overfit.

This is a **small, measurable augmentation** of Titouan's already sophisticated query bank, not a full new model.

## Reddit source 2: small segmentation / tiling is not magic (T1)
- [Reddit r/computervision, May 2026](https://www.reddit.com/r/computervision/comments/1t7zyya/how_to_detect_small_object_from_far_away_using/): resolution, IoU discontinuity at small regions, 2-stage crop/refine and HRNet-style high-res maps.
- [Reddit r/computervision, Oct 2023](https://www.reddit.com/r/computervision/comments/17aytqw): simple overlapping image crops with coordinate remapping; strong caution that single-image super-resolution does **not** recreate lost evidence.
- Team has already measured shifted full-image **PixelSeal decoder** (16 shifts) with no improvement; **do not repeat**. Candidate38 already full-res features. If resources free, only consider a *localizer f_x2* gated crop-and-refine on <64px rectangles, not the decoder; OOF and reserve with same-sign improvement, paired CI95 excluding0. This is lower priority than T2.

## Reddit source 3: SSD unlearning needs retain data, unlike EAC (T3)
- [Reddit r/MachineLearning, Aug 2023 SSD author](https://www.reddit.com/r/MachineLearning/comments/15slvx8/r_fast_machine_unlearning_without_retraining/) and [LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/16skep1): selective synaptic dampening uses *forget* vs *retain* Fisher importance.
- EAC provides **no retain set** and scores a ViT against hidden gold per-image distributions; this makes direct SSD a low-confidence first choice. n2 nullspace/ridge and retain anchors have already lost locally/publicly; do not repeat by default.

## Truly new (paper, NOT Reddit): 2026 TREW-inspired geometry reweighting for T3, NO training before OOF gate

[ICML 2026 **Unlearning Isn't Forgetting**](https://proceedings.mlr.press/v306/ebrahimpour-boroojeny26a.html), [author code](https://github.com/CrowdDynamicsLab/ICML_TREW), uses **inter-class similarity** to tilt distributions over *other classes* toward what independent retraining would predict. Original paper addresses class unlearning and uses label information; EAC is **unlabeled instance unlearning**, so **do not claim direct transfer**.

**Candidate adaptation**: Build 100×100 cosine class geometry from *original* ViT `head.weight`. For each forgotten image with fixed pseudo-label y, reweight *conditional distribution of the remaining 99 classes* using `q_beta(c|x,y) ∝ q_RDq(c|x,y) exp(beta cos(head_y,head_c))`. **Keep mass of y fixed** so m/s CE doesn't move in the target by construction. Compare betas {−2,−1,−0.5,0,+0.5,+1,+2} on raw per-fold OOF probabilities (T=2); β=0 is exact baseline, no fabricated per-teacher predictions.

Added [`t3_trew_geometry_oof.py`](../tracks/t3-unlearning/experiments/t3_trew_geometry_oof.py): teacher split K/2 vs K/2, image fit/evaluation split, report holdout paired-bootstrap improvement in teacher-consistency KL. **Caveat**: a shared teacher bias is *invisible* to this cross-validation; passing is merely consistency among pseudo-gold teachers, not proof of true gold proximity. Only propose a ViT checkpoint if heldout gain >0 and bootstrap CI excludes0, AND Adem agrees, AND m/s and retain/copies/checker pass. If no signal, STOP.
- The algorithm uses `orig.argmax` **pseudo-labels, not true labels**; true labels unavailable; original TREW score claims do not transfer.
- Algorithmic primitive spot-checked with synthetic NumPy data: β0 exactly equals source q (max error ~1.1e-16); with no shared bias between teachers, β0 was selected as expected. **No real checkpoint / EAC score**.
- Expected upside unknown. **Do not crowd out the in-flight RDq 16/24 OOF teacher experiments**.

## Three biggest NO-GO findings
1. Reddit has **no verified direct EAC top1 recipe**; an unvalidated adaptation cannot be sold as winning strategy.
2. **T2 public score should not dictate hidden partition** without checking final CSV selection and source evidence; prior statements of fixed 30% T2 were incorrect.
3. **T3 retain is not the bottleneck** per Adem's latest G10 probes; more anchor runs without new evidence have bad expected value.

Source authority order: **challenge specification > measurements in private EAC repo > peer-reviewed papers with code > Reddit discussions**. All source claims separated from our hypotheses.
