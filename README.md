# Concrete Compressive Strength Analysis

Reproduce the CSE5104 Project 1 experiments with custom batch gradient descent,
scikit-learn least-squares benchmarks, and statsmodels OLS inference.

## Files

- `Part1_Run_All.py`: complete analysis, exported from the Part 1 notebook.
- `Concrete_Data.xls`: the 1,030-observation Concrete Compressive Strength dataset.
- `requirements.txt`: exact analysis package versions used for the reported run.
- `README.md`: setup and execution instructions.

## Reproduce all results

Use Python 3.13.1. From a terminal:

```sh
git clone https://github.com/Haoyun253/Concrete-Compressive-Strength-Analysis.git
cd Concrete-Compressive-Strength-Analysis
python -m pip install -r requirements.txt
python Part1_Run_All.py
```

Run the commands from the repository root so the script can find the spreadsheet.
No command-line arguments or notebook frontend are required. Dependencies are
installed explicitly; the analysis does not install packages automatically.
If plot windows open, close each window to continue to the next experiment.
For a run without windows, set `MPLBACKEND=Agg` before starting Python.
In PowerShell, use `$env:MPLBACKEND = 'Agg'`; in a POSIX shell, use
`MPLBACKEND=Agg python Part1_Run_All.py`.

## Experimental protocol

The Excel header is excluded from sample numbering. Rows 501–630 are the 130
test samples (`iloc[500:630]`); the other 900 rows are used for training.
The original order is preserved. StandardScaler is fitted on training inputs
only. The target remains in MPa. The log representation is `log1p(X)`.

Gradient descent uses zero initialization, at most 50,000 updates, and an
absolute consecutive-loss tolerance of `1e-10`. Unless supplied explicitly to
`gd`, the learning rate is `0.9 / L`, where `L` is the largest eigenvalue of
`2 * A.T @ A / n` and `A` includes the intercept. No random seed is needed.

## Outputs and expected checks

The script prints environment versions, dataset checks, performance metrics,
convergence diagnostics, coefficient comparisons, and p-values. It writes:

- `Q3_1_Loss_Curves.png`
- `Q5_1_Loss_Curves.png`
- `Q5_3_Model_Comparison.png`
- `Q6_2_Coefficients.png`
- `Q7_1_P_Values.png`

Both raw and standardized multivariate gradient descent runs achieve positive
training R²; four of eight univariate runs do so. The raw and standardized
optimal models agree in their predictions. The optimal log model gives:

| Metric | Training | Testing |
| --- | ---: | ---: |
| MSE (MPa²) | 55.832418 | 54.384907 |
| R² | 0.798450 | 0.756959 |

Small last-digit differences may occur across numerical libraries. Generated
figures and duplicate notebook/Q2 copies are omitted to keep this repository
compact; the full-run script includes all their analysis code.
