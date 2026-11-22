classes du jeu a oublier : 45
| reservoir | images | classes couvertes /100 (>=1 image) | >=10 images | part des images dans les 45 classes | confiance orig (max-prob T=1) moy / mediane | part max-prob > 0,9 | entropie moy (nats) | precision si etiquette connue |
|---|---|---|---|---|---|---|---|---|
| images a oublier exactes | 3500 | 45 | 45 | 1.000 | 1.000 / 1.000 | 1.000 | 0.002 | 1.000 |
| melanges moitie au plus (c) : tous | 6000 | 100 | 94 | 0.664 | 0.748 / 0.800 | 0.374 | 0.773 | - |
|   dont mixup 2-3 images | 2000 | 92 | 67 | 0.692 | 0.758 / 0.815 | 0.391 | 0.729 | - |
|   dont coupe moitie-moitie | 2000 | 97 | 68 | 0.710 | 0.770 / 0.834 | 0.414 | 0.703 | - |
|   dont mosaique 2x2 | 2000 | 99 | 60 | 0.590 | 0.716 / 0.740 | 0.319 | 0.886 | - |
| melanges de t3-methodes (cache) | 3000 | 100 | 72 | 0.742 | 0.786 / 0.863 | 0.451 | 0.635 | - |
| copies degradees blur_strong | 3500 | 85 | 49 | 0.951 | 0.919 / 0.997 | 0.773 | 0.239 | 0.849 |
| copies degradees color_strong | 3500 | 98 | 66 | 0.801 | 0.840 / 0.949 | 0.583 | 0.473 | 0.532 |
| genere (a) flux_i2i_60 | 900 | 49 | 45 | 0.994 | 0.979 / 1.000 | 0.942 | 0.060 | 0.974 |
| genere (a) flux_t2i | 900 | 96 | 37 | 0.734 | 0.817 / 0.915 | 0.523 | 0.542 | 0.326 |
| genere (a) sdxl_i2i_30 | 900 | 46 | 45 | 0.999 | 0.995 / 1.000 | 0.986 | 0.015 | 0.997 |
| genere (a) sdxl_i2i_50 | 900 | 53 | 45 | 0.991 | 0.981 / 1.000 | 0.942 | 0.056 | 0.972 |
| genere (a) sdxl_i2i_70 | 900 | 74 | 45 | 0.936 | 0.919 / 0.996 | 0.766 | 0.230 | 0.796 |
| genere (a) sdxl_i2i_85 | 900 | 89 | 44 | 0.857 | 0.872 / 0.964 | 0.636 | 0.367 | 0.568 |
| genere (a) sdxl_ipa_60 | 900 | 85 | 43 | 0.879 | 0.877 / 0.981 | 0.647 | 0.338 | 0.624 |
| genere (a) sdxl_t2i | 900 | 91 | 40 | 0.758 | 0.817 / 0.911 | 0.510 | 0.532 | 0.347 |
| COCO val2017 (b) | 5000 | 98 | 86 | 0.491 | 0.734 / 0.771 | 0.337 | 0.814 | - |
