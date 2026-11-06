"""Plot foci per nucleus for each treatment condition.

Reads a table with one column per condition and one row per cell (empty cells
allowed), draws every cell as a dot, overlays the median and interquartile
range, and marks significant differences.

Run:  python plot_foci.py                      (uses data/simulated_foci_counts.csv)
      python plot_foci.py my_counts.csv        (any table with the same layout)
Out:  figures/foci_per_nucleus.png  (300 dpi, for slides)
      figures/foci_per_nucleus.pdf  (vector, for papers/posters)
      printed summary table and statistics
"""

import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "data" / "simulated_foci_counts.csv"
FIG_DIR = HERE / "figures"

# Okabe-Ito colour-blind-safe palette, one colour per condition (in CSV order).
COLOURS = ["#7F7F7F", "#0072B2", "#E69F00", "#D55E00"]

# Comparisons to test and draw as brackets: (condition A, condition B).
COMPARISONS = [("WT", "EZH2i"), ("WT", "PARPi"), ("PARPi", "EZH2i + PARPi"),
               ("WT", "EZH2i + PARPi")]


def load_counts(path: Path) -> dict[str, np.ndarray]:
    """Return {condition: array of foci counts}, dropping empty cells."""
    table = pd.read_csv(path)
    return {col: table[col].dropna().to_numpy(dtype=float) for col in table.columns}


def sina_offsets(values: np.ndarray, width: float, rng: np.random.Generator) -> np.ndarray:
    """Horizontal offsets that spread dots in proportion to the local data density.

    Dots sit wide where many cells share a value and narrow in the tails, so the
    cloud of dots has the shape of a violin plot while every cell stays visible.
    """
    if len(values) < 3 or np.ptp(values) == 0:
        return rng.uniform(-width, width, len(values))
    density = stats.gaussian_kde(values)(values)
    return rng.uniform(-1, 1, len(values)) * width * density / density.max()


def stars(p: float) -> str:
    """Convert a p-value into the usual star notation."""
    for cutoff, label in [(1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (0.05, "*")]:
        if p < cutoff:
            return label
    return "ns"


def compare(counts: dict[str, np.ndarray]) -> list[tuple[str, str, float]]:
    """Two-sided Mann-Whitney U tests, Holm-corrected for multiple comparisons."""
    raw = [stats.mannwhitneyu(counts[a], counts[b]).pvalue for a, b in COMPARISONS]
    adjusted = _holm(raw)
    return [(a, b, p) for (a, b), p in zip(COMPARISONS, adjusted)]


def _holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def plot(counts: dict[str, np.ndarray], results: list[tuple[str, str, float]], out_stem: Path) -> None:
    names = list(counts)
    rng = np.random.default_rng(0)          # fixed jitter -> identical figure every run

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6, 5))

    for i, (name, colour) in enumerate(zip(names, COLOURS)):
        y = counts[name]
        x = i + sina_offsets(y, 0.32, rng)
        ax.scatter(x, y, s=16, color=colour, alpha=0.55, edgecolor="none", zorder=2)

        q1, median, q3 = np.percentile(y, [25, 50, 75])
        ax.plot([i - 0.28, i + 0.28], [median] * 2, color="black", lw=2.2, zorder=3)
        ax.plot([i, i], [q1, q3], color="black", lw=1.2, zorder=3)
        for q in (q1, q3):
            ax.plot([i - 0.08, i + 0.08], [q] * 2, color="black", lw=1.2, zorder=3)

    # Additive expectation: what the combination would give if the two drugs
    # simply added their individual effects on top of WT.
    med = {n: np.median(counts[n]) for n in names}
    additive = med["EZH2i"] + med["PARPi"] - med["WT"]
    combo = names.index("EZH2i + PARPi")
    ax.plot([combo - 0.4, combo + 0.4], [additive] * 2, ls="--", lw=1.2, color="#444444", zorder=4)
    ax.annotate("additive\nexpectation", xy=(combo + 0.4, additive), xytext=(combo + 0.48, additive),
                va="center", ha="left", fontsize=8.5, color="#444444", annotation_clip=False)

    # Significance brackets, stacked above the data.
    top = max(v.max() for v in counts.values())
    step = top * 0.065
    for level, (a, b, p) in enumerate(sorted(results, key=lambda r: abs(names.index(r[1]) - names.index(r[0])))):
        x1, x2 = names.index(a), names.index(b)
        h = top + step * (level + 0.8)
        ax.plot([x1, x1, x2, x2], [h - step * 0.25, h, h, h - step * 0.25], color="black", lw=0.9)
        ax.text((x1 + x2) / 2, h, stars(p), ha="center", va="bottom", fontsize=10)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n.replace(' + ', ' +\n')}\n(n = {len(counts[n])})" for n in names], fontsize=10)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(0, top + step * (len(results) + 1.2))
    ax.set_ylabel("Foci per nucleus")
    ax.set_title("EZH2 inhibition amplifies PARPi-induced foci", fontsize=12, weight="bold", loc="left")
    fig.text(0.01, 0.01, "SIMULATED DATA (teaching example). Dot = one nucleus; bar = median; whiskers = IQR.\n"
             "Two-sided Mann-Whitney U, Holm-corrected: "
             "**** p < 0.0001.", fontsize=7.5, color="#555555")

    fig.tight_layout(rect=(0, 0.05, 0.94, 1))
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300)
    print(f"saved {out_stem}.png and .pdf")


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    counts = load_counts(csv_path)

    summary = pd.DataFrame({n: {"n": len(v), "median": np.median(v), "mean": v.mean(),
                                "sd": v.std(ddof=1)} for n, v in counts.items()}).T
    print(summary.round(1), "\n")

    results = compare(counts)
    for a, b, p in results:
        print(f"{a:>14} vs {b:<14} p_adj = {p:.2e}  {stars(p)}")

    plot(counts, results, FIG_DIR / "foci_per_nucleus")


if __name__ == "__main__":
    main()
