"""Generate the SIMULATED foci-count table used for the class plot.

These numbers are not real measurements. They are drawn at random so that
the plot looks like a typical DNA damage foci experiment:

    condition        mean foci per nucleus
    WT               ~10
    EZH2i            ~20
    PARPi            ~35
    EZH2i + PARPi    ~60

Foci counts are whole numbers and spread out more as the mean grows, so they
are drawn from a negative binomial distribution (a Poisson with extra
cell-to-cell variability). A fixed random seed makes the file identical
every time this script runs.

Run:  python scripts/make_simulated_data.py
Out:  data/simulated_foci_counts.csv  (one column per condition, one row per cell)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 46050                      # fixed seed -> same "data" every run
MEANS = {"WT": 10, "EZH2i": 20, "PARPi": 35, "EZH2i + PARPi": 60}
DISPERSION = 8                    # lower = more cell-to-cell variability
N_RANGE = (50, 60)                # cells per condition, chosen at random in this range
OUT = Path(__file__).resolve().parent.parent / "data" / "simulated_foci_counts.csv"


def draw_counts(rng: np.random.Generator, mean: float, n: int) -> np.ndarray:
    """Draw n foci counts with the given mean from a negative binomial."""
    p = DISPERSION / (DISPERSION + mean)    # numpy's (n, p) parameterisation
    return rng.negative_binomial(DISPERSION, p, size=n)


def main() -> None:
    rng = np.random.default_rng(SEED)
    columns = {}
    for condition, mean in MEANS.items():
        n = rng.integers(N_RANGE[0], N_RANGE[1] + 1)
        columns[condition] = pd.Series(draw_counts(rng, mean, n), dtype="Int64")

    # Columns have different lengths; shorter ones are padded with empty cells.
    table = pd.DataFrame(columns)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(table)} rows)")
    print(table.describe().round(1))


if __name__ == "__main__":
    main()
