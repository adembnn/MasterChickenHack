# Pierre Fernandez : les sources qu'il cite, et qui le cite

Données Semantic Scholar (auteur 2147013351, Meta AI et Inria), relevées le 21 novembre 2026.

## Profil

16 papiers indexés, 14 332 citations, h-index 12. Attention au trompe-l'œil : 12 437 de ces
citations vont à DINOv2, et 771 à deux projets voisins (Cookbook of SSL 418, Seamless 353). Le
corpus watermarking réel pèse 1 355 citations, soit environ 9 % du profil. C'est un chercheur de
watermarking cité surtout pour du self-supervised learning.

| citations | année | papier |
|---|---|---|
| 12 437 | 2023 | DINOv2: Learning Robust Visual Features without Supervision |
| 527 | 2023 | The Stable Signature: Rooting Watermarks in Latent Diffusion Models |
| 418 | 2023 | A Cookbook of Self-Supervised Learning |
| 353 | 2023 | Seamless: Multilingual Expressive and Streaming Speech Translation |
| 196 | 2024 | Proactive Detection of Voice Cloning with Localized Watermarking (AudioSeal) |
| 189 | 2021 | Watermarking Images in Self-Supervised Latent Spaces |
| 120 | 2023 | Three Bricks to Consolidate Watermarks for Large Language Models |
| 97 | 2024 | Watermark Anything with Localized Messages |
| 79 | 2024 | Watermarking Makes Language Models Radioactive |
| 72 | 2024 | Video Seal: Open and Efficient Video Watermarking |
| 32 | 2024 | Latent Watermarking of Audio Generative Models |
| 24 | 2023 | Functional Invariants To Watermark Large Transformers |
| 11 | 2022 | Active Image Indexing |
| 8 | 2024 | Evaluation of Security of ML-based Watermarking: Copy and Removal Attacks |
| 0 | 2026 | Foundations and Frontiers of Watermarking (survey) |

## Les sources qu'il cite, remontées sur 11 papiers

782 références uniques au total. Les plus récurrentes, et ce qu'elles apportent :

- **Emerging Properties in Self-Supervised Vision Transformers** (DINO, 2021, 7 occurrences) et
  **An Image is Worth 16x16 Words** (ViT, 5) : l'encodeur. Ses extracteurs sont des ViT, ses
  watermarks vivent dans des features auto-supervisées avant d'être appris.
- **HiDDeN: Hiding Data With Deep Networks** (2018, 6) : la recette originale, embedder et extractor
  entraînés de bout en bout avec des transformations au milieu. Tout le reste en descend.
- **The Stable Signature** (6, auto-citation) et **Watermarking Images in Self-Supervised Latent
  Spaces** (4, auto-citation) : ses deux marches.
- **Responsible Disclosure of Generative Models Using Scalable Fingerprinting** (2020, 6) et
  **Artificial Fingerprinting for Generative Models** (2020, 4) : le cadre conceptuel, le
  fingerprinting comme divulgation responsable, qu'il reprend pour WOUAF et Stable Signature.
- **WOUAF: Weight Modulation for User Attribution and Fingerprinting** (2023, 5) : le concurrent
  direct sur l'attribution par utilisateur, modulation des poids du décodeur.
- **Tree-Ring Watermarks** (2023, 4) : l'approche sémantique, dans le sampling, qu'il présente comme
  la famille opposée à la sienne.
- **Distortion Agnostic Deep Watermarking** (2020, 4), **MBRS** (2021), **ReDMark** (2018),
  **ROMark** (2019), **Are Classification DNNs Good for Blind Image Watermarking** (2020) : la
  génération intermédiaire, qui règle chacune un axe (compression JPEG, transformations composées,
  robustesse adversariale).
- **Digital Watermarking and Steganography** (2009, 5) : le socle théorique, capacité, robustesse,
  imperceptibilité.
- Outils de mesure : **LPIPS** (5), **SSIM** (5), **PSNR implicite**, **COCO** (4), **Adam** (4),
  **Latent Diffusion** (4).

Deux choses à noter. Un, il ne cite presque pas la littérature de sécurité offensive, sauf son
propre papier de 2024 sur les attaques de copie et de retrait. Deux, la théorie ancienne arrive par
deux entrées seulement : l'ouvrage de référence de 2009 pour le socle (capacité, robustesse,
imperceptibilité), et l'article de 2005 de Cayre, Fontaine et Furon pour la distinction entre
robustesse et sécurité, qu'il reprend mot pour mot à la slide 51 de son talk. Les articles de
tatouage de 1998 à 2005, eux, ne sont pas cités, ce qui explique que son vocabulaire soit celui de
l'apprentissage profond et pas celui de la théorie du signal.

Attention aussi à une attribution : WOUAF n'est pas de lui, c'est Kim, Min, Patel, Cheng et Yang
(ASU et Intel), CVPR 2024. Il le cite comme concurrent sur l'attribution par utilisateur.

## Qui le cite

Échantillon de 802 papiers citants sur ses cinq travaux watermark les plus cités :

- ArXiv 233, revues et conférences non renseignées 112, NeurIPS 28, ICLR 27, INTERSPEECH 16,
  ICML 13, ICASSP 11, CVPR 11, ECCV 8, TMLR 8, ICCV 10, WACV 8, IEEE TMM 9, IEEE TCSVT 10,
  IEEE Access 8, Expert Systems with Applications 8.

Lecture : le centre de gravité est NeurIPS, ICLR et ICML, pas les conférences de sécurité. Ce sont
des papiers d'apprentissage qui traitent le watermark comme une couche du modèle, pas des attaquants
qui cherchent à le casser. Le volume INTERSPEECH et ICASSP vient d'AudioSeal.

Conséquence pour le hackathon : les organisateurs viennent du camp offensif (SprintML, CISPA) et
invitent le camp défensif (Meta FAIR). Ils ne testent pas la même chose. Un jury CISPA demandera
« est-ce que ton watermark résiste à un attaquant qui a le gradient », là où les papiers de Fernandez
optimisent la robustesse aux transformations classiques. C'est exactement la distinction qu'il fait
lui-même à la slide 51 de son talk, robustesse contre sécurité.

## Papiers manquants dans l'index Semantic Scholar

Trois travaux récents de 2026 ne sont pas encore indexés avec leurs citations : TextSeal, TextSeal2,
Watermark Basins, DistSeal, SyncSeal, We Can Hide More Bits, Forging, Rephrasing, Benchmark
Contamination, Geometric Synchronization. Le décompte ci-dessus sous-estime donc l'impact réel de
2025 et 2026.
