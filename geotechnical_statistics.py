"""
Statistical analysis and visualization of seven geotechnical parameters.

Expected parameters:
    CBR, G, S, F, PI, MDD, OMC

Input:
    Excel workbook (default: SPSS2.xlsx)

Outputs (saved to analysis_results/):
    - geotechnical_statistics_7params.csv
    - statistical_barplots_7params.png
    - distribution_histograms_7params.png
    - correlation_heatmap_7params.png
    - pairplot_7params.png

Install dependencies:
    pip install -r requirements.txt

Run:
    python geotechnical_statistics.py
    python geotechnical_statistics.py --input path/to/data.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew


EXPECTED_COLUMNS = {
    "CBR": [
        "CBR",
        "CBR (Y1)",
        "The California Bearing Ratio, CBR (Y1)",
    ],
    "G": ["G", "G (X1)", "The gravel content, G (X1)"],
    "S": ["S", "S (X2)", "The sand content, S (X2)"],
    "F": ["F", "F (X3)", "The fine content, F (X3)"],
    "PI": [
        "PI",
        "PI (X4)",
        "PI (X6)",
        "The plasticity index, PI (X6)",
        "The plasticity index, PI (X4)",
    ],
    "MDD": ["MDD", "MDD (X7)", "The maximum dry density, MDD (X7)"],
    "OMC": ["OMC", "OMC (X8)", "The optimum moisture content, OMC (X8)"],
}

HISTOGRAM_COLORS = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#96CEB4",
    "#FFEAA7",
    "#DDA0DD",
    "#90CAF9",
]


def match_and_standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Find recognized parameter columns and rename them to standard names."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    matched = {}
    for standard_name, aliases in EXPECTED_COLUMNS.items():
        for alias in aliases:
            if alias in df.columns:
                matched[standard_name] = alias
                break

    if not matched:
        available = ", ".join(map(str, df.columns))
        raise ValueError(
            "No recognized parameter columns were found. "
            f"Available columns: {available}"
        )

    # Keep one source column per parameter and use consistent standard names.
    selected = df[[source for source in matched.values()]].copy()
    rename_map = {source: standard for standard, source in matched.items()}
    selected.rename(columns=rename_map, inplace=True)

    for column in selected.columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    return selected


def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive statistics for each parameter, ignoring missing values."""
    records = []

    for column in df.columns:
        values = df[column].dropna().to_numpy(dtype=float)
        if values.size == 0:
            warnings.warn(f"Skipping statistics for {column}: no numeric data.")
            continue

        mode_values = pd.Series(values).mode()
        records.append(
            {
                "Parameter": column,
                "Mean": np.mean(values),
                "Median": np.median(values),
                "Mode": mode_values.iloc[0] if not mode_values.empty else np.nan,
                "SD": np.std(values, ddof=1) if values.size > 1 else np.nan,
                "Skewness": skew(values, bias=False) if values.size > 2 else np.nan,
                # Fisher's definition: normal distribution has kurtosis equal to 0.
                "Kurtosis": kurtosis(values, fisher=True, bias=False)
                if values.size > 3
                else np.nan,
                "Range": np.ptp(values),
                "Minimum": np.min(values),
                "Maximum": np.max(values),
                "Count": values.size,
            }
        )

    if not records:
        raise ValueError("No usable numeric observations were found.")

    return pd.DataFrame(records).set_index("Parameter")


def plot_statistical_summary(stats_df: pd.DataFrame, output_dir: Path) -> None:
    """Create horizontal bar charts for mean, standard deviation, skewness, kurtosis."""
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.titlecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )

    metrics = ["Mean", "SD", "Skewness", "Kurtosis"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "Statistical Summary of Geotechnical Parameters",
        fontsize=16,
        fontweight="bold",
    )

    for ax, metric in zip(axes.flat, metrics):
        values = stats_df[metric].astype(float)
        valid = values.dropna()

        if valid.empty:
            ax.set_visible(False)
            continue

        low, high = valid.min(), valid.max()
        if np.isclose(low, high):
            bar_colors = [plt.get_cmap("coolwarm")(0.5)] * len(values)
        else:
            norm = plt.Normalize(low, high)
            cmap = plt.get_cmap("coolwarm")
            bar_colors = [
                cmap(norm(value)) if pd.notna(value) else (0.7, 0.7, 0.7, 1)
                for value in values
            ]

        bars = ax.barh(
            stats_df.index,
            values.fillna(0),
            color=bar_colors,
            edgecolor="black",
            height=0.75,
        )
        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel(metric, fontweight="bold")
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)

        if metric in {"Skewness", "Kurtosis"}:
            ax.axvline(0, color="gray", linestyle="--", linewidth=1)

    fig.tight_layout()
    fig.savefig(
        output_dir / "statistical_barplots_7params.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_histograms(df: pd.DataFrame, output_dir: Path, bins: int = 15) -> None:
    """Plot a count histogram for each parameter with mean and median reference lines."""
    n_parameters = len(df.columns)
    n_cols = 3
    n_rows = int(np.ceil(n_parameters / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for index, (parameter, series) in enumerate(df.items()):
        values = series.dropna()
        ax = axes[index]
        ax.hist(
            values,
            bins=bins,
            alpha=0.9,
            color=HISTOGRAM_COLORS[index % len(HISTOGRAM_COLORS)],
            edgecolor="white",
            linewidth=1.2,
        )

        if not values.empty:
            ax.axvline(
                values.mean(),
                color="#D62728",
                linestyle="--",
                label=f"Mean: {values.mean():.3f}",
            )
            ax.axvline(
                values.median(),
                color="#1F77B4",
                linestyle="--",
                label=f"Median: {values.median():.3f}",
            )

        ax.set_title(parameter, fontweight="bold")
        ax.set_xlabel(parameter)
        ax.set_ylabel("Count")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

    for ax in axes[n_parameters:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(
        output_dir / "distribution_histograms_7params.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot the Pearson correlation matrix for the recognized parameters."""
    corr = df.corr(numeric_only=True)
    size = max(9, 1.5 * len(corr.columns))
    fig, ax = plt.subplots(figsize=(size, size))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.8,
        annot_kws={"size": 10},
        cbar_kws={"label": "Pearson correlation coefficient"},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap of Geotechnical Parameters", fontweight="bold")
    fig.tight_layout()
    fig.savefig(
        output_dir / "correlation_heatmap_7params.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_pairplot(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot pairwise relationships and univariate distributions."""
    sns.set_style("whitegrid")
    pairplot = sns.pairplot(
        df,
        diag_kind="hist",
        corner=True,
        plot_kws={"alpha": 0.75, "s": 28, "color": "#2E86AB"},
        diag_kws={"color": "#FF6B6B", "edgecolor": "white"},
    )
    pairplot.fig.suptitle(
        "Pairwise Relationships of Geotechnical Parameters",
        y=1.02,
        fontsize=15,
        fontweight="bold",
    )
    pairplot.savefig(
        output_dir / "pairplot_7params.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(pairplot.fig)


def analyze_dataset(file_path: str | Path, output_dir: str | Path = "analysis_results"):
    """Load an Excel workbook, calculate statistics, and save tables and plots."""
    file_path = Path(file_path)
    output_dir = Path(output_dir)

    if not file_path.is_file():
        raise FileNotFoundError(f"Excel file not found: {file_path.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_excel(file_path)
    print(f"Dataset loaded: {file_path}")
    print(f"Original dataset shape: {raw_df.shape}")

    df = match_and_standardize_columns(raw_df)
    print(f"Parameters detected: {', '.join(df.columns)}")

    stats_df = calculate_statistics(df)
    stats_df.to_csv(output_dir / "geotechnical_statistics_7params.csv")
    print("\nDescriptive statistics:")
    print(stats_df.round(6).to_string())

    plot_statistical_summary(stats_df, output_dir)
    plot_histograms(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_pairplot(df, output_dir)

    print(f"\nAnalysis complete. Results saved in: {output_dir.resolve()}")
    return stats_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate and visualize statistics for geotechnical parameters."
    )
    parser.add_argument(
        "--input",
        default="SPSS2.xlsx",
        help="Path to the input Excel workbook (default: SPSS2.xlsx).",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_results",
        help="Directory for CSV and figure outputs (default: analysis_results).",
    )
    args = parser.parse_args()

    try:
        analyze_dataset(args.input, args.output_dir)
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    main()
