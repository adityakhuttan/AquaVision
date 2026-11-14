# Classifier local test (CPU, 13 Nov 2026): inference-only fixes + unknown-species check

**Setup:** F4K cloned locally and the split rebuilt with the baseline flags. Per-class train/val/test
counts are identical, and **both models reproduce their Kaggle test confusion matrices exactly** (0 cells
differ), so this is the same test set as the Kaggle runs. Every option was chosen on **VAL** and then
scored once on **TEST** (2,737 crops). The open-set check uses 1,523 Brackish fish crops (55 clips, species
not in F4K).

## Test results
| Config | top-1 | macro-F1 | clip-split 16 | mid | tail | errors | rare→fish_01 | fish_01→rare |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.9868 | 0.9208 | 0.9404 | 0.928 | 0.879 | 36 | 19 | 6 |
| aug1 (previous demo model) | 0.9872 | 0.9399 | 0.9553 | 0.946 | 0.909 | 35 | 7 | 18 |
| aug1 + flip | 0.9872 | 0.9304 | 0.9593 | 0.951 | 0.883 | 35 | 9 | 17 |
| aug1 + prior correction | = aug1 (val picked tau = 0) | | | | | | | |
| **ensemble 0.25 baseline + 0.75 aug1, + flip (picked on val)** | **0.9887** | **0.9428** | **0.9615** | **0.956** | 0.908 | **31** | 11 | 11 |

Paired comparison of the ensemble vs aug1 on the same crops: 10 fixed, 6 broken, McNemar p = 0.45,
macro-F1 difference 95% CI [−0.023, +0.028]. **Not statistically significant**; this test set is too
small to separate them.

## Unknown-species cut-off (max softmax; a threshold keeping a fraction of correct VAL predictions)
| Model | Threshold | F4K test kept | Accuracy on kept | Brackish rejected |
|---|---|---|---|---|
| ensemble + flip | 0.96694 (keep 99 %) | 0.982 | 0.9952 | 0.609 |
| **ensemble + flip** | **0.99827 (keep 98 %)** | **0.965** | **0.9966** | **0.982** |
| aug1 alone | 0.97566 (keep 99 %) | 0.977 | 0.9936 | 0.834 |
| **aug1 alone** | **0.99790 (keep 98 %)** | **0.959** | **0.9958** | **0.965** |

Without a cut-off, Brackish fish are labelled fish_02 (738) or fish_11 (673) with a median confidence
of 0.93.

Re-run: `scripts/` (`make_brackish_crops.py` → `cls_predict.py` → `cls_analyse.py`); full numbers are in
`cls_local_results.json`.
