#!/usr/bin/env python3
"""Create a publication-style ROC/AUC plot from a wide CSV file.

Expected CSV layout::

    sample_id,true_label,Model_A,Model_B
    S001,0,0.12,0.31
    S002,1,0.88,0.67

The ``true_label`` column must contain binary labels. Every other column
except the optional ``sample_id`` column is treated as a model probability.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


class CSVFormatError(ValueError):
    """Raised when the input CSV is not a valid ROC/AUC table."""


def _load_dependencies():
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.metrics import auc, roc_curve
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency. Install the toolkit requirements with "
            "`python -m pip install -r ../requirements.txt`."
        ) from exc
    return auc, plt, pd, roc_curve


def read_csv_file(path: str | Path) -> tuple[Any, list[str]]:
    """Read and validate a wide CSV, returning the table and model columns."""
    _, _, pd, _ = _load_dependencies()
    input_path = Path(path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    if input_path.suffix.lower() != ".csv":
        raise CSVFormatError("Please provide a .csv input file.")

    try:
        data = pd.read_csv(input_path)
    except Exception as exc:
        raise CSVFormatError(f"Could not read CSV {input_path}: {exc}") from exc

    if "true_label" not in data.columns:
        raise CSVFormatError("The CSV must contain a `true_label` column.")
    if data.empty:
        raise CSVFormatError("The CSV contains no data rows.")

    labels = pd.to_numeric(data["true_label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise CSVFormatError("`true_label` must contain only binary values 0 and 1.")
    if labels.nunique() != 2:
        raise CSVFormatError("ROC/AUC requires both negative (0) and positive (1) labels.")
    data = data.copy()
    data["true_label"] = labels.astype(int)

    excluded = {"sample_id", "true_label"}
    model_columns = [column for column in data.columns if column not in excluded]
    if not model_columns:
        raise CSVFormatError("No model score columns were found.")

    for model in model_columns:
        scores = pd.to_numeric(data[model], errors="coerce")
        if scores.isna().any():
            raise CSVFormatError(f"Model column {model!r} contains non-numeric or missing values.")
        if not scores.between(0, 1).all():
            raise CSVFormatError(f"Model column {model!r} must contain probabilities in [0, 1].")
        data[model] = scores.astype(float)

    return data, model_columns


def calculate_auc_summary(data: Any, model_columns: list[str]) -> list[dict[str, float | str]]:
    """Calculate one AUC value per model, preserving CSV column order."""
    _, _, _, roc_curve = _load_dependencies()
    from sklearn.metrics import roc_auc_score

    labels = data["true_label"]
    summary: list[dict[str, float | str]] = []
    for model in model_columns:
        fpr, tpr, _ = roc_curve(labels, data[model])
        summary.append({
            "model": model,
            "auc": float(roc_auc_score(labels, data[model])),
            "n_thresholds": len(fpr),
        })
    return summary


# Muted, distinguishable colors; the first four match the original four-model style.
PALETTE = [
    "#214A55",
    "#A65D43",
    "#6E8B74",
    "#6B7280",
    "#C9A227",
    "#7B5EA7",
    "#4C9BC7",
    "#C2185B",
    "#8C564B",
    "#2E7D32",
]
LINE_STYLES = ["-", "--", ":", "-."]


def default_title(model_count: int) -> str:
    """Return a title that matches the number of model columns."""
    if model_count == 1:
        return "ROC Curve"
    return f"ROC Curves for {model_count} Models"


def _summary_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_summary.csv")


def make_plot(
    data: Any,
    model_columns: list[str],
    summary: list[dict[str, float | str]],
    title: str | None,
    output_path: str | Path,
) -> Path:
    """Render and save the ROC plot and AUC summary CSV."""
    auc, plt, pd, roc_curve = _load_dependencies()
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.labelweight": "normal",
        "axes.titlesize": 17,
        "axes.titleweight": "normal",
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.linewidth": 1.5,
    })

    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    fig.subplots_adjust(left=0.18, bottom=0.16, right=0.96, top=0.90)

    for index, row in enumerate(summary):
        model = str(row["model"])
        fpr, tpr, _ = roc_curve(data["true_label"], data[model])
        curve_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=PALETTE[index % len(PALETTE)],
            # Change line style only after all colors are used, so curves stay distinct.
            linestyle=LINE_STYLES[(index // len(PALETTE)) % len(LINE_STYLES)],
            linewidth=2.0,
            alpha=0.95,
            label=f"{model} (AUC = {curve_auc:.3f})",
            zorder=3,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle=(0, (4, 3)),
        color="#7A7A7A",
        linewidth=1.2,
        label="Random classifier",
        zorder=1,
    )
    if title is None:
        title = default_title(len(summary))
    ax.set_title(title, fontweight="normal", pad=14)
    ax.set_xlabel("False Positive Rate", fontweight="normal")
    ax.set_ylabel("True Positive Rate", fontweight="normal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    ax.grid(False)
    ax.tick_params(axis="both", length=5, width=1.3)
    ax.legend(loc="lower right", frameon=False, fontsize=12)

    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary_path = _summary_path(output_path)
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(summary_path, index=False, columns=["model", "auc", "n_thresholds"])
    return output_path


def run(
    input_path: str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> tuple[Path, Path]:
    """Read input data, calculate AUC, and save the plot and summary."""
    input_path = Path(input_path).expanduser()
    data, model_columns = read_csv_file(input_path)
    summary = calculate_auc_summary(data, model_columns)
    output = Path(output_path).expanduser() if output_path else input_path.with_name(f"{input_path.stem}_roc.png")
    make_plot(data, model_columns, summary, title, output)
    return output, _summary_path(output)


def main(
    csv_file: str | Path | None = None,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> int:
    """Run from a notebook or command line, prompting for a missing CSV path."""
    if csv_file is None:
        entered = input("CSV filename (path or filename in this folder): ").strip()
        if not entered:
            print("Error: no CSV filename was provided.", file=sys.stderr)
            return 1
        csv_file = entered

    try:
        output, summary = run(csv_file, output_path, title)
    except (CSVFormatError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"PNG saved to: {output}")
    print(f"AUC summary saved to: {summary}")
    return 0


def cli_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path, nargs="?", help="Input wide-format CSV file")
    parser.add_argument("-o", "--output", type=Path, help="PNG output path")
    parser.add_argument("--title", help="Plot title; defaults to 'ROC Curves for <n> Models'")
    args = parser.parse_args()
    return main(args.csv_file, args.output, args.title)


if __name__ == "__main__":
    raise SystemExit(cli_main())
