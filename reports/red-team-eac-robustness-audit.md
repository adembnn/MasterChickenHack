# Red-team audit — MasterChickenHack T1/T2/T3 (22 Nov 2026)

> **Threat-model hypotheses, not successful attacks.** Public results from GitHub issues, not live private leaderboard. No T2 endpoint calls, GPU tests, SSH access, credential collection, or contest submission here.

| Track | Latest confirmed public | Local evidence | Most exposed failure mode |
|---|---:|---|---|
| T1 | **0.7700**, submission 2010, candidate24 ([#3](https://github.com/adembnn/EAC/issues/3)) | Candidate24 reserve 0.7536; PixelSeal reserve ~0.648, StegaStamp ~0.842 | Watermark already decoded globally; small clean holes/irregular pasted objects leave weak local evidence |
| T2 | **0.60**, submission 1967; separation of ep17/18 in 2052 also **0.60** ([#4](https://github.com/adembnn/EAC/issues/4)) | Ref 1.00 achieved but some fresh cross-family twin classifiers ~0.58–0.64 | Text distribution changes from persona/system/decode look like new weights; weak LoRA/quantization looks like same weights |
| T3 | **0.3273**, submission 2031, PG_RDt24_60_l915c ([#5](https://github.com/adembnn/EAC/issues/5)) | Gold-proxy aggressive experiment 0.0804; redistill +0.026 at same 60% fraction vs f60 | Loss moments on exact forget samples can improve without matching gold per-sample, especially on hidden altered copies |

## T1: attacker vs localization detector

**Authorized synthetic threat tests only.** In all cases record *localization* IoU separately for the marked and clean class; watermark payload decode success is not localization success. Dataset 512×512, score evaluated at 128×128 (4×4 source pixels/cell); aggregate by method × edit type × clean-hole size, not just one average.

1. Tiny content-following edits: replace 0.5%, 1%, 2%, and 5% of a watermarked image with original clean pixels using irregular object-shaped masks, including narrow outlines. Detection of the payload may remain high while the actual hole is invisible to the localizer. Metric: mIoU_clean and pixel-map AUC, by edit size. Compare against candidate24 and exact 128-grid rule.
2. Nonrectangular/object transplant: composite an irregular watermarked object on a clean image with smooth alpha boundaries and varying edge softness. Tests whether the proposed SAM vote erroneously expands the mark into the clean background. Metric: IoU_watermarked and boundary F1, and object-vs-rectangle group scores.
3. Mild re-encoding/coordinate shifts AFTER compositing: JPEG (Q 75/85/95), resized 0.85–1.15 then restored, 1–3px translation and crop/pad. Threat: fragile re-embedding canvas registration or residual cues; measure changes in inferred frames and mIoU, while preserving perceptual similarity.
4. Generative-edit stress (if permissible locally): image inpainting/removal + small object addition on same sources. Strong but **out-of-distribution to the EAC test unless replicated by task rules**; report separately, don't tune official choice solely to it.
5. Attribution caveat: successful watermark *detection* is insufficient for provenance: altered image can carry the original watermark in untouched pixels even when the incriminating object is new.

**Defense/next step**: Most headroom is PixelSeal local signal, not more thresholds. Do NOT waste time repeating decoder-position CAM/SAM full masks/black-hat, already measured negative. Trial content-conditioned hole pairs and native high-resolution features **only on honest OOF folds**; compare on same held-out images against c24, with paired CI. T1 team submit rule against c22 reserve .7514 is +.020 → .7714; candidate24 .7536 is below the gate.

## T2: attacker vs model-identity fingerprinting

Challenge identity is checkpoint + quantization + LoRA; **system prompt, decode strategy and watermark are nuisance variables, NOT identity**. Public metric is MCC, 90% comes from *same model* among within-family pairs. False 'same' and false 'different' both hurt.

1. **False split (same weights, altered behavior)**: deployment changes system prompt/persona, temperature/top-p, or decoding watermark while keeping weights constant; plain text similarity will split the same model. Explicitly reproduced in practice refs; ep07/14/27 were identified as top-p=.95, T=1 copies. Score against known-weight twins with strong nuisance combinations.
2. **False merge (different weights, nearly identical text)**: mild LoRA (particularly 0.3), int8/FP8 under noisy sampling, or quantized copies. Their top-1 outputs are often the same at 25 responses; a language-style detector groups distinct models. LoRA0.3 was essentially invisible to current ref LR; current int8 twin concordance 0/3.
3. **Confounded teacher**: learning an endpoint-specific LLR from a disguised endpoint can learn the system prompt *rather than weights*, yielding confident wrong model labels. The team's twin tests already observed this, including 0/2 under a system-prompt corpus.
4. **Leaderboard ambiguity**: public score 0.60 after both grouping and separating ep17/ep18 DOES NOT identify the true model equality because that pair might not be evaluated publicly; do not conclude shared or distinct weights.

**Defense/next step**: fix a confusion matrix over *fresh* within-family twins with independent random seeds, system/persona/T/top-p nuisance, known quant/LoRA; cross-fit on disjoint prompts; calibrate a null of fake same-model endpoints; require signal supported by a reader trained on a DIFFERENT endpoint. No additional ep API calls unless Titouan explicitly approves.

## T3: attacker vs purported forgetting

The goal is behavioral match to gold retrained without the 3500 images, not merely wrong top1 or high CE. Scored components multiply (exact+altered forget loss moments, per-image KL at T=2, retain KL+accuracy), utility gate ≥70%.

1. **Near-duplicate recovery**: a model tuned on exact 3500 images may respond differently on 1–3px crops, mild rotation/blur, color transforms and semantically equivalent object crops. If the original class confidence reappears on such variants, the forgetting is brittle; measure per-image rebound probability and CE, not only mean.
2. **Memorization audit**: compare original vs submitted vs independently retrained/matched OOF reference on forget and *independent nonmembers*, with membership-inference ROC/AUC and calibrated likelihood under matched augmentations. High forget loss alone is no certificate of removal. Access to genuine nonmembers/gold may be limited; state that.
3. **Optimization to proxy failure**: gold_blend55 local proxy looked attractive but scored **0.0804**, demonstrating serious simulator/domain mismatch; any 'attack' that targets different image transformations or gold diversity can reveal false security. Audit T3 with holdout transformations and disjoint retain proxies.
4. **Target class mistakes**: redistill changed the WRONG class at the same f60 target sharpness and improved 0.3009→**0.3273**. This directly shows choosing *which* wrong class matters more than simply producing low accuracy. Evaluate per-sample KL to multiple independent OOF teachers, and consistency under copies.
5. **Utility gate cliff**: a candidate could look excellent on forget while crossing the 70% control threshold and scoring zero. Calibrate retain accuracy on independent, distribution-matched samples with confidence interval, not merely COCO.

**Defense/next step**: prioritize independent OOF multi-seed redistill, agreement by sample and exact-vs-augmented copy verification while preserving retain proxy/gate; don't duplicate Adem's running multi-seed jobs. Head-only ridge repairs may restore forget knowledge: test per-example rebound before submission.

## Operational attack surface (not a claim of compromise)

The conversation previously included an EAC API key and a private SSH key was subsequently uploaded. **Rotate/revoke these credentials if still active**, remove them from any broad logs or shared artifacts, and use per-agent least-privilege credentials. Publishing paths, fingerprints and public SSH keys in GitHub isn't itself equivalent to publishing private keys. No secret material is included here.

## Suggested deliverable for the team

A single CSV per track with one row per attack category and columns: `attack_family`, `method`, `seed`, `strength`, `n`, `clean_metric`, `attacked_metric`, `delta`, `paired_CI95`, `notes`. Keep exact comparisons paired; avoid changing the baseline between adversarial tests. Prevent surprises at the 70% private cutoff by prioritizing hidden-set generalization, not public-only score.
