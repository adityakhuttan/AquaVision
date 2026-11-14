"""Quantization chart: per model, CPU latency and file size (separate panels, single axis each)."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

q = json.load(open(sys.argv[1])); out = Path(sys.argv[2])
INK, MUTED, GRID = "#0C1B1F", "#41544F", "#D5DFDC"
COL = {"PyTorch FP32": "#8FA39E", "ONNX FP32": "#0E9E93", "ONNX INT8": "#0A5C58"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlecolor": "#0A5C58", "axes.titlesize": 10})
rows = [("Species classifier (224 px)", q["classifier"], ("pytorch_fp32", "onnx_fp32", "onnx_int8"), "macro_f1", "macro-F1"),
        ("Detector (640 px)", q["detector"], ("pytorch_fp32", "onnx_fp32", "onnx_int8_head_fp32"), "box_mAP50", "box mAP50")]
names = ["PyTorch FP32", "ONNX FP32", "ONNX INT8"]
fig, axes = plt.subplots(2, 2, figsize=(8.4, 4.4), gridspec_kw={"hspace": 0.75, "wspace": 0.55})
for r, (title, d, keys, acc, accname) in enumerate(rows):
    for c, (metric, unit) in enumerate((("latency_ms", "CPU latency per image (ms)"), ("size_mb", "File size (MB)"))):
        ax = axes[r, c]; vals = [d[k][metric] for k in keys]
        bars = ax.barh(range(3), vals, color=[COL[n] for n in names], height=0.6, edgecolor="white", linewidth=2)
        for i, (b, k) in enumerate(zip(bars, keys)):
            lab = f"{vals[i]:.0f}" if metric == "latency_ms" else f"{vals[i]:.1f}"
            if metric == "latency_ms": lab += f"   ({accname} {d[k][acc]:.3f})"
            ax.text(b.get_width() + max(vals) * 0.02, i, lab, va="center", fontsize=8.3, color=INK)
        ax.set_yticks(range(3)); ax.set_yticklabels(names); ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * (1.85 if metric == "latency_ms" else 1.3)); ax.set_xlabel(unit, fontsize=8.5)
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        if c == 0: ax.set_title(title, loc="left")
fig.text(0.01, -0.04, "Laptop CPU (i5-1135G7), batch 1, ONNX Runtime. Detector INT8 keeps the output head in FP32; "
         "quantizing the head too drops mAP to 0.", fontsize=7.8, color=MUTED)
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); print("wrote", out)
