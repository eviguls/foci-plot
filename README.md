# foci-plot

Plot DNA damage foci per nucleus across treatment conditions with Python:
every cell shown as a dot, median and interquartile range, significance tests,
and an additive-expectation line for drug combinations.

## Input data

A CSV file with **one column per condition** and **one row per cell**. Each
number is the foci count of one nucleus. Columns can have different numbers of
cells; leave the extra cells empty.

```
WT,EZH2i,PARPi,EZH2i + PARPi
9,21,30,58
12,17,41,66
...
```

- The **first column is the control**: every other condition is compared to it.
- A column named **`A + B`**, where `A` and `B` are also columns (e.g.
  `EZH2i + PARPi`), is treated as a combination. It is also compared to `A` and
  `B`, and a dashed line shows the additive expectation:
  median(A) + median(B) - median(control).

Put your file at `data/foci_counts.csv`, or pass any path on the command line.
`data/` and `figures/` are not tracked by git.

## How to run

```bash
git clone https://github.com/eviguls/foci-plot.git
cd foci-plot
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate foci-plot
python plot_foci.py                                         # data/foci_counts.csv
python plot_foci.py path/to/counts.csv --title "Your message here"
```

Options: `--title` (plot title; make it your message), `--ylabel` (default
"Foci per nucleus").

Output: `figures/<csv name>.png` (300 dpi, for slides) and `.pdf` (vector, for
posters and papers), plus a printed summary table (n, median, mean, SD) and the
adjusted p-values.

## How the plot is built (`plot_foci.py`)

| Element | Why |
|---|---|
| Every cell is a dot, spread sideways by data density ("sina" plot) | Shows all the data and its shape, not just an average |
| Black bar = median, whiskers = interquartile range | Foci counts are skewed; median/IQR are not pulled by a few extreme cells |
| n under each label | Reader knows how many cells each condition has |
| Dashed "additive expectation" line on combinations | Shows whether the combination is more than the two treatments added together |
| Two-sided Mann-Whitney U tests, Holm-corrected | Non-parametric (no normality assumption); correction for multiple tests |
| Colour-blind-safe (Okabe-Ito) palette, PNG 300 dpi + vector PDF | Readable on slides and in print |

Note: statistics treat each cell as independent. For publication, use
biological replicates (e.g. 3 independent experiments) as the unit of analysis.
