"""Inference-only fixes for the species classifier, chosen on VAL, reported once on TEST.
Variants: prior correction (logit adjustment), flip-TTA, baseline+aug ensemble. Plus paired tests vs aug1
and an 'unknown species' confidence threshold checked on Brackish crops (species outside F4K)."""
import sys, csv, json, itertools, random
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

NPZ, RES, OUTJ = sys.argv[1], sys.argv[2], sys.argv[3]          # probs.npz, results dir, output json
d = np.load(NPZ)
names = list(d["names_base"]); assert names == list(d["names_aug"])
K = len(names); idx = {n: i for i, n in enumerate(names)}
y = {s: np.array([idx[c] for c in d[f"y_{s}"]]) for s in ("val", "test")}

def prior(path):
    rows = {r["class"]: int(r["train_final"]) for r in csv.DictReader(open(path))}
    p = np.array([rows[n] for n in names], float); return p / p.sum()
pri = {"base": prior(f"{RES}/cls_baseline/class_report.csv"), "aug": prior(f"{RES}/cls_aug1/class_report.txt")}
rep = {r["class"]: r for r in csv.DictReader(open(f"{RES}/cls_aug1/class_report.txt"))}
clip = [idx[c] for c in rep if rep[c]["split_mode"] == "group"]
real = {c: int(r["train_real"]) for c, r in rep.items()}
grp = {"head": [idx[c] for c in real if real[c] >= 1000], "mid": [idx[c] for c in real if 100 <= real[c] < 1000],
       "tail": [idx[c] for c in real if real[c] < 100]}

def logp(m, s, flip):
    p = d[f"{m}_{s}_plain"]
    if flip: p = (p + d[f"{m}_{s}_flip"]) / 2
    return np.log(p + 1e-9)

def scores(s, w, tau, flip):
    return w * (logp("base", s, flip) - tau * np.log(pri["base"])) + \
           (1 - w) * (logp("aug", s, flip) - tau * np.log(pri["aug"]))

# ---- sanity: do the local predictions reproduce the Kaggle test confusion matrices?
for m, f in (("base", "cls_baseline/test_confusion.json"), ("aug", "cls_aug1/test_confusion.txt")):
    kag = json.load(open(f"{RES}/{f}")); assert kag["labels"] == names
    loc = confusion_matrix(y["test"], d[f"{m}_test_plain"].argmax(1), labels=list(range(K)))
    diff = int(np.abs(loc - np.array(kag["matrix"])).sum())
    print(f"[check] {m}: local test confusion vs Kaggle -> {diff} cells differ "
          f"(local errors {int((loc.sum() - np.trace(loc)))}, Kaggle {int(np.sum(kag['matrix']) - np.trace(kag['matrix']))})")

mf1 = lambda t, p, lab=None: f1_score(t, p, labels=lab if lab is not None else list(range(K)), average="macro", zero_division=0)

# ---- grid search on VAL only
grid = []
for w, tau, flip in itertools.product((0, .25, .5, .75, 1), (0, .25, .5, .75, 1, 1.25, 1.5), (0, 1)):
    pv = scores("val", w, tau, flip).argmax(1)
    grid.append((mf1(y["val"], pv), (pv == y["val"]).mean(), w, tau, flip))
grid.sort(key=lambda g: (-round(g[0], 4), -g[1], abs(g[3]), g[2] not in (0, 1)))
best = grid[0]; print("top 5 on VAL (macroF1, top1, w_base, tau, flip):")
for g in grid[:5]: print("  ", tuple(round(x, 4) if isinstance(x, float) else x for x in g))

named = {"baseline": (1, 0, 0), "aug1 (current demo)": (0, 0, 0), "aug1 + flip": (0, 0, 1),
         "aug1 + prior corr (val-tuned tau)": (0, max((g for g in grid if g[2] == 0 and g[4] == 0), key=lambda g: g[0])[3], 0),
         "ensemble 50/50 + flip": (.5, 0, 1), "SELECTED on val": (best[2], best[3], best[4])}

def summary(pred, s="test"):
    t = y[s]
    out = {"top1": float((pred == t).mean()), "macro_f1": float(mf1(t, pred)),
           "macro_f1_clip16": float(mf1(t, pred, clip)), "errors": int((pred != t).sum()),
           "rare_as_fish01": int(((pred == 0) & (t != 0)).sum()), "fish01_as_rare": int(((t == 0) & (pred != 0)).sum())}
    for g, lab in grp.items(): out[f"macro_f1_{g}"] = float(mf1(t, pred, lab))
    return out

res = {"val_grid_best": {"w_base": best[2], "tau": best[3], "flip": best[4], "val_macro_f1": best[0]}, "test": {}}
preds = {k: scores("test", *v).argmax(1) for k, v in named.items()}
print(f"\n{'TEST':36s} top1   mF1    clip16  head   mid    tail   err  rare->01 01->rare")
for k, p in preds.items():
    sm = summary(p); res["test"][k] = {**sm, "config": dict(zip(("w_base", "tau", "flip"), named[k]))}
    print(f"{k:36s} {sm['top1']:.4f} {sm['macro_f1']:.4f} {sm['macro_f1_clip16']:.4f} {sm['macro_f1_head']:.3f}  "
          f"{sm['macro_f1_mid']:.3f}  {sm['macro_f1_tail']:.3f}  {sm['errors']:3d}  {sm['rare_as_fish01']:5d}  {sm['fish01_as_rare']:5d}")

# ---- paired comparison SELECTED vs aug1 on the same test crops
a, b, t = preds["aug1 (current demo)"], preds["SELECTED on val"], y["test"]
only_a, only_b = int(((a == t) & (b != t)).sum()), int(((a != t) & (b == t)).sum())
from math import comb
n = only_a + only_b; k = min(only_a, only_b)
p_mcn = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
rng = random.Random(0); diffs = []
for _ in range(2000):
    ii = np.array([rng.randrange(len(t)) for _ in range(len(t))])
    diffs.append(mf1(t[ii], b[ii]) - mf1(t[ii], a[ii]))
diffs.sort()
res["paired_vs_aug1"] = {"aug1_right_selected_wrong": only_a, "selected_right_aug1_wrong": only_b,
                         "mcnemar_p": p_mcn, "macro_f1_diff_ci95": [diffs[50], diffs[1949]]}
print(f"\nSELECTED vs aug1 (paired): fixed {only_b}, broke {only_a}, McNemar p={p_mcn:.3f}, "
      f"macro-F1 diff 95% CI [{diffs[50]:+.4f}, {diffs[1949]:+.4f}]")

# ---- unknown-species threshold: max softmax of the SELECTED scores; tuned on F4K VAL, checked on Brackish
def conf(s):
    z = scores(s, best[2], best[3], best[4]); z = np.exp(z - z.max(1, keepdims=True)); z /= z.sum(1, keepdims=True)
    return z.max(1), z.argmax(1)
cv_, pv_ = conf("val"); ct, pt = conf("test"); cb, pb = conf("brackish")
res["unknown_threshold"] = []
print(f"\n{'threshold':>9s} {'F4K val kept':>12s} {'F4K test kept':>13s} {'test acc kept':>13s} {'Brackish rejected':>17s}")
for keep in (0.99, 0.98, 0.95, 0.90):
    thr = float(np.quantile(cv_[pv_ == y['val']], 1 - keep))     # keep `keep` of CORRECT val predictions
    kt = ct >= thr
    row = {"keep_target": keep, "threshold": thr, "val_kept": float((cv_ >= thr).mean()), "test_kept": float(kt.mean()),
           "test_acc_kept": float((pt[kt] == y["test"][kt]).mean()), "brackish_rejected": float((cb < thr).mean())}
    res["unknown_threshold"].append(row)
    print(f"{thr:9.3f} {row['val_kept']:12.3f} {row['test_kept']:13.3f} {row['test_acc_kept']:13.4f} {row['brackish_rejected']:17.3f}")
print("Brackish crops: median max-confidence", round(float(np.median(cb)), 3), "| F4K test median", round(float(np.median(ct)), 3))
print("species most often given to Brackish fish:",
      sorted(((int((pb == i).sum()), names[i]) for i in range(K)), reverse=True)[:5])
res["brackish_n"] = int(len(cb))
cm = confusion_matrix(y["test"], preds["SELECTED on val"], labels=list(range(K)))
res["selected_test_confusion"] = {"labels": names, "matrix": cm.tolist()}
json.dump(res, open(OUTJ, "w"), indent=1); print("\n->", OUTJ)
