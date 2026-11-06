"""Plot foci per nucleus for each treatment condition.

Reads a table with one column per condition and one row per cell (empty cells
allowed), draws every cell as a dot, overlays the median and interquartile
range, and marks significant differences.

The FIRST column is treated as the control: every other condition is tested
against it. A column named "A + B", where A and B are also columns (e.g.
"EZH2i + PARPi"), is treated as a combination: it is also tested against A and
B, and a dashed line marks the additive expectation, median(A) + median(B) -
median(control).

Run:  python plot_foci.py                              (uses data/foci_counts.csv)
      python plot_foci.py my_counts.csv --title "MCF7, 6 Gy, 1 h"
Out:  figures/<csv name>.png  (300 dpi, for slides)
      figures/<csv name>.pdf  (vector, for papers/posters)
      printed summary table and statistics
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "data" / "foci_counts.csv"
FIG_DIR = HERE / "figures"

# Okabe-Ito colour-blind-safe palette; grey for the control, then colours in order.
COLOURS = ["#7F7F7F", "#0072B2", "#E69F00", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#F0E442"]


def load_counts(path: Path) -> dict[str, np.ndarray]:
    """Return {condition: array of foci counts}, dropping empty cells."""
    table = pd.read_csv(path)
    return {col.strip(): table[col].dropna().to_numpy(dtype=float) for col in table.columns}


def find_combinations(names: list[str]) -> dict[str, tuple[str, str]]:
    """Return {"A + B": (A, B)} for every column that combines two other columns."""
    combos = {}
    for name in names:
        parts = [p.strip() for p in name.split("+")]
        if len(parts) == 2 and all(p in names for p in parts):
            combos[name] = (parts[0], parts[1])
    return combos


def choose_comparisons(names: list[str]) -> list[tuple[str, str]]:
    """Every condition vs the control (first column), plus each combination vs its parts."""
    control = names[0]
    pairs = [(control, n) for n in names[1:]]
    for combo, parts in find_combinations(names).items():
        pairs += [(part, combo) for part in parts if part != control]
    return pairs


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


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction for multiple comparisons."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def compare(counts: dict[str, np.ndarray]) -> list[tuple[str, str, float]]:
    """Two-sided Mann-Whitney U tests, Holm-corrected."""
    pairs = choose_comparisons(list(counts))
    raw = [stats.mannwhitneyu(counts[a], counts[b]).pvalue for a, b in pairs]
    return [(a, b, p) for (a, b), p in zip(pairs, holm(raw))]


def plot(counts: dict[str, np.ndarray], results: list[tuple[str, str, float]],
         out_stem: Path, title: str, ylabel: str) -> None:
    names = list(counts)
    rng = np.random.default_rng(0)          # fixed jitter -> identical figure every run

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(1.2 + 1.2 * len(names), 5))

    for i, name in enumerate(names):
        y = counts[name]
        x = i + sina_offsets(y, 0.32, rng)
        ax.scatter(x, y, s=16, color=COLOURS[i % len(COLOURS)], alpha=0.55, edgecolor="none", zorder=2)

        q1, median, q3 = np.percentile(y, [25, 50, 75])
        ax.plot([i - 0.28, i + 0.28], [median] * 2, color="black", lw=2.2, zorder=3)
        ax.plot([i, i], [q1, q3], color="black", lw=1.2, zorder=3)
        for q in (q1, q3):
            ax.plot([i - 0.08, i + 0.08], [q] * 2, color="black", lw=1.2, zorder=3)

    # Additive expectation for each combination: what it would give if the two
    # treatments simply added their individual effects on top of the control.
    med = {n: np.median(v) for n, v in counts.items()}
    for combo, (a, b) in find_combinations(names).items():
        additive = med[a] + med[b] - med[names[0]]
        i = names.index(combo)
        ax.plot([i - 0.4, i + 0.4], [additive] * 2, ls="--", lw=1.2, color="#444444", zorder=4)
        ax.annotate("additive\nexpectation", xy=(i + 0.42, additive), va="center", ha="left",
                    fontsize=8.5, color="#444444", annotation_clip=False)

    # Significance brackets, stacked above the data, shortest first.
    top = max(v.max() for v in counts.values())
    step = top * 0.085
    ordered = sorted(results, key=lambda r: abs(names.index(r[1]) - names.index(r[0])))
    for level, (a, b, p) in enumerate(ordered):
        x1, x2 = names.index(a), names.index(b)
        h = top + step * (level + 0.8)
        ax.plot([x1, x1, x2, x2], [h - step * 0.15, h, h, h - step * 0.15], color="black", lw=0.9)
        ax.text((x1 + x2) / 2, h, stars(p), ha="center", va="bottom", fontsize=10)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n.replace(' + ', ' +' + chr(10))}\n(n = {len(counts[n])})" for n in names],
                       fontsize=10)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(0, top + step * (len(results) + 1.2))
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=12, weight="bold", loc="left")
    fig.text(0.01, 0.01, "Dot = one nucleus; bar = median; whiskers = IQR.\n"
             "Two-sided Mann-Whitney U, Holm-corrected: * p<0.05, ** p<0.01, "
             "*** p<0.001, **** p<0.0001.", fontsize=7.5, color="#555555")

    fig.tight_layout(rect=(0, 0.05, 0.94, 1))
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300)
    print(f"saved {out_stem}.png and .pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV,
                        help="table with one column per condition, one row per cell")
    parser.add_argument("--title", default="", help="plot title (your message)")
    parser.add_argument("--ylabel", default="Foci per nucleus", help="y-axis label")
    args = parser.parse_args()

    counts = load_counts(args.csv)
    summary = pd.DataFrame({n: {"n": len(v), "median": np.median(v), "mean": v.mean(),
                                "sd": v.std(ddof=1)} for n, v in counts.items()}).T
    print(summary.round(1), "\n")

    results = compare(counts)
    for a, b, p in results:
        print(f"{a:>14} vs {b:<14} p_adj = {p:.2e}  {stars(p)}")

    plot(counts, results, FIG_DIR / args.csv.stem, args.title, args.ylabel)


if __name__ == "__main__":
    main()
