# T3, 22 November 2026 — retained-drift nullspace abliteration after PG-redistillation

## Executive decision
**Status: experimental hypothesis / not GPU-tested / not submitted.**
As of issue #5 comment 7072351735 (02:39 Paris), public T3 best is **0.3942**, submission #2531, the seed-1 checkpoint `SOUPs1`, same method as `PG_RDt24r_70_l1016c` (0.3888). The old Yassir handoff's **0.777** is a self-designed local loss-moment proxy, **not EAC**, and its original public checkpoint scored **0.327**. Do not rank T3 experiments by 0.777 or by CIFAR alone.

## External sources checked
1. Arditi et al. (NeurIPS 2024), *Refusal in Language Models Is Mediated by a Single Direction*, https://arxiv.org/abs/2406.11717 . The original technique surgically projects model activations/weights away from an identified behavioral direction. **Its results concern refusal in LLMs, not ViT image unlearning**; transferring it directly is not supported.
2. Open implementation with norm/retain protection: https://github.com/jwest33/abliterator . The useful idea is to constrain an unwanted update with nullspace projections; it does not imply that one-dimensional refusal signals exist for our arbitrary forget images.
3. Poppi et al. (ICPR 2024), *Unlearning Vision Transformers without Retaining Data via Low-Rank Decompositions*, https://iris.unimore.it/handle/11380/1350486 . Relevant architecture/retain-free setting, but primarily **class** unlearning, not our example-level gold-KL.
4. Tang et al. (2026), *Unmerge: Efficient Machine Unlearning via Task Arithmetic*, https://arxiv.org/abs/2609.38895 . Forget/retain activation subspaces and small low-rank corrections; **the paper itself warns that random instance removal may lack a separable subspace**. Therefore stop if the proxy subspace does not separate.

## New test: ABL-PG nullspace instead of raw model abliteration

**Never erase the original model's weights or semantic features.** Start from the team's *actual best* fully validated PG candidate `W_best` and original `W_orig`.

- Use only the 3,500 allowed forget images and already generated mixup/cutmix/patch-mix images from them (EAC `t3-methodes/cache/mixes.npy`; not a true retain set).
- Under the original frozen ViT, obtain `pre_logits` (768-dimensional) on both groups. L2-normalize rows, form `C_mix - C_forget`, and diagonalize it. Positive eigenvectors are directions with more energy in pseudo-retain than forget.
- Edit ONLY the classification head delta, `D = W_best.head - W_orig.head`. For top-r positive eigenvectors U, compute `D' = D - alpha (D U) U^T`; new head is `W_orig.head + D'`. ALL OTHER WEIGHTS/BIASES remain exactly `W_best`. Grid: r in {4,8,16,32}, alpha in {.25,.5}.
- First cheap screen: ratio `||F D'^T|| / ||F D^T|| >= .95` (do not erase most forget response) and reduction `1 - ||Mix D'^T||/||Mix D^T|| >= .02` (small, measurable pseudo-retain benefit). **These are NOT the EAC forgetting/retention scores**.
- Save no more than 3 candidates. Validate strict architecture, float32, finite weights, <450 MB. No external label or T2 request.

**Implementation** (PR #34 branch): `tracks/t3-unlearning/experiments/t3_retain_nullspace.py`.

### Execution on an authorized free H200

```bash
python tracks/t3-unlearning/experiments/t3_retain_nullspace.py \
  --orig ~/eac/t3/model.pt \
  --best ~/eac/t3/t3-methodes/candidates/PG_RDt24r_70_l1016c_s1.pt \
  --forget ~/eac/t3/forget_images.npz \
  --retain ~/eac/t3/t3-methodes/cache/mixes.npy \
  --t3base ~/eac/t3 --out ~/runs/t3_nullspace
```

The example path must be verified; the current winning file may be elsewhere or newer. Do not start any job or occupy GPUs without checking nvidia-smi, running processes and #2 allocation.

For 1–3 screened outputs, run the team's `dash.py --sd name=/full/path.pt`, separately evaluate exact 3500 + 4 mild + strong color/blur/rotation, holdout imitation/reweighting, and KL on proxy controls. **The actual best** `SOUPs1` has exact CE m/s ~4.869/4.553 and mild ~4.238/4.173 (issue #5). Do not accept a candidate if this shape worsens, copy robustness collapses, or retain/gate risk increases. A proxy retain gain is not validation. Cross-check on independent heldout forget folds with a scorer not used for tuning.

**GO/NO-GO:** If real EAC-local metrics improve *clearly*, give candidate path, sha256, exact+copies m/s, KL, gate margin and independent checker verdict to issue #5. Only Adem's submission orchestrator may call the T3 submit endpoint (decision of 01:00, `CLAUDE.md`). Otherwise stop, no blind submissions.

## Why this is genuinely distinct
- Not the 60/70/80% fraction sweep; those were tested.
- Not the 4-seed soup already queued by Adem.
- Not the existing per-layer scalar delta sweep in PR #34: this removes **specific retain-dominant head directions** of an existing good delta rather than globally scaling layer groups.
- Does not train on or import external datasets, avoiding the T3 data-scope problem.

## Known weaknesses
- Mixes are only a crude surrogate for control/retain; minimizing proxy logit norm can hurt the true gate or gold KL.
- Source paper's class-level results cannot justify instance-level performance here.
- If the head delta does not encode selective memorization, orthogonal projection can do nothing; the 3,500 samples may have no positive separable directions.
- Public 30% score variance and unknown hidden 70% mean a 0.002 gain does not establish generalization.
