#!/usr/bin/env python3
"""Create publication-style confusion matrices from true/predicted labels.

Example:
    python confusion_matrix.py
    python confusion_matrix.py confusion_matrix_demo_5class.csv

The input CSV must contain ``true_label`` and ``predicted_label`` columns.
The class count is inferred from the data, so the same protocol works for
4-, 5-, 8-, or other n-class classification problems.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


DEFAULT_TRUE_COLUMN = "true_label"
DEFAULT_PREDICTED_COLUMN = "predicted_label"


def prompt_user_configuration(
    input_func: Callable[[str], str] = input,
) -> tuple[Path, str, str]:
    """Interactively collect the CSV filename and the two label columns."""
    entered_filename = input_func("CSV filename (path or filename in this folder): ").strip()
    if not entered_filename:
        raise ValueError("No CSV filename was provided.")

    entered_true_column = input_func(
        f"True-label column name [{DEFAULT_TRUE_COLUMN}]: "
    ).strip()
    entered_predicted_column = input_func(
        f"Predicted-label column name [{DEFAULT_PREDICTED_COLUMN}]: "
    ).strip()
    return (
        Path(entered_filename).expanduser(),
        entered_true_column or DEFAULT_TRUE_COLUMN,
        entered_predicted_column or DEFAULT_PREDICTED_COLUMN,
    )


def read_label_rows(
    input_path: str | Path,
    true_column: str = DEFAULT_TRUE_COLUMN,
    predicted_column: str = DEFAULT_PREDICTED_COLUMN,
    *,
    require_matching_label_sets: bool = True,
) -> list[dict[str, str]]:
    """Read and validate true/predicted labels from a CSV file."""
    path = Path(input_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in (true_column, predicted_column) if column not in fieldnames]
        if missing:
            raise ValueError(
                f"Missing required column(s): {', '.join(missing)}. "
                f"Available columns: {', '.join(fieldnames) or '(none)'}"
            )

        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            true_value = row.get(true_column)
            predicted_value = row.get(predicted_column)
            true_label = "" if true_value is None else str(true_value).strip()
            predicted_label = "" if predicted_value is None else str(predicted_value).strip()
            if not true_label or not predicted_label:
                raise ValueError(f"Empty true or predicted label at CSV row {row_number}.")
            rows.append({"true_label": true_label, "predicted_label": predicted_label})

    if not rows:
        raise ValueError("The input CSV contains no data rows.")

    true_labels = {row["true_label"] for row in rows}
    predicted_labels = {row["predicted_label"] for row in rows}
    if require_matching_label_sets and true_labels != predicted_labels:
        raise ValueError(
            "True and predicted label sets do not match: "
            f"true-only={sorted(true_labels - predicted_labels)!r}, "
            f"predicted-only={sorted(predicted_labels - true_labels)!r}. "
            "Provide the complete class order with `labels=[...]` in the notebook or "
            "--labels on the command line when a class is absent from predictions."
        )
    return rows


def infer_labels(rows: Iterable[dict[str, str]]) -> list[str]:
    """Infer labels from true-label order, then append unseen predictions."""
    labels: list[str] = []
    seen: set[str] = set()
    materialized_rows = list(rows)
    for key in ("true_label", "predicted_label"):
        for row in materialized_rows:
            label = row[key]
            if label not in seen:
                labels.append(label)
                seen.add(label)
    if len(labels) < 2:
        raise ValueError("At least two distinct classes are required for a confusion matrix.")
    return labels


def build_confusion_matrix(
    rows: Iterable[dict[str, str]], labels: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw counts and row-normalized percentages for ``labels``."""
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("labels must contain at least two unique class names.")

    label_to_index = {label: index for index, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)), dtype=int)
    for row in rows:
        true_label = row["true_label"]
        predicted_label = row["predicted_label"]
        if true_label not in label_to_index or predicted_label not in label_to_index:
            raise ValueError(
                f"Found label outside the requested class order: {true_label!r}, {predicted_label!r}."
            )
        counts[label_to_index[true_label], label_to_index[predicted_label]] += 1

    row_totals = counts.sum(axis=1, keepdims=True)
    percentages = np.divide(
        counts * 100.0,
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals != 0,
    )
    return counts, percentages


def format_annotation(count: int, percentage: float) -> str:
    """Format one cell as count plus row percentage."""
    return f"{int(count)}\n({percentage:.1f}%)"


def _figure_size(number_of_classes: int) -> float:
    """Scale the square canvas without making small matrices look oversized."""
    return min(12.0, max(6.4, 5.5 + 0.42 * number_of_classes))


def _font_sizes(number_of_classes: int) -> tuple[float, float, float]:
    """Return tick, axis, and cell annotation sizes for n-class matrices."""
    tick_size = max(8.5, 15.0 - 0.55 * max(number_of_classes - 5, 0))
    axis_size = max(10.0, tick_size)
    annotation_size = max(7.0, 13.0 - 0.65 * max(number_of_classes - 5, 0))
    return tick_size, axis_size, annotation_size


def make_plot(
    counts: np.ndarray,
    percentages: np.ndarray,
    labels: list[str],
    output_prefix: str | Path,
    *,
    title: str | None = None,
    font_family: str = "DejaVu Sans",
    dpi: int = 600,
) -> tuple[Path, Path]:
    """Render and save a publication-style PNG and vector PDF."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if counts.shape != (len(labels), len(labels)):
        raise ValueError("counts must be a square matrix matching labels.")
    if percentages.shape != counts.shape:
        raise ValueError("percentages must have the same shape as counts.")

    output_prefix = Path(output_prefix).expanduser()
    # Accept "result.png" or "result.pdf" as a prefix, but never use with_suffix():
    # it would truncate names containing dots, such as "results.v2_confusion_matrix".
    if output_prefix.suffix.lower() in {".png", ".pdf"}:
        output_prefix = output_prefix.with_suffix("")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_name(f"{output_prefix.name}.png")
    pdf_path = output_prefix.with_name(f"{output_prefix.name}.pdf")

    number_of_classes = len(labels)
    tick_size, axis_size, annotation_size = _font_sizes(number_of_classes)
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": tick_size,
            "axes.labelsize": axis_size,
            "axes.titlesize": axis_size + 1,
            "axes.linewidth": 1.2,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure_size = _figure_size(number_of_classes)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size), constrained_layout=False)
    fig.subplots_adjust(
        left=0.22 if number_of_classes > 7 else 0.18,
        bottom=0.23 if number_of_classes > 5 else 0.18,
        right=0.86,
        top=0.90 if title else 0.96,
    )

    maximum = max(int(counts.max()), 1)
    image = ax.imshow(
        counts,
        cmap="Blues",
        vmin=0,
        vmax=maximum,
        interpolation="nearest",
        aspect="equal",
    )

    positions = np.arange(number_of_classes)
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    if number_of_classes > 5:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Predicted label", labelpad=10)
    ax.set_ylabel("True label", labelpad=10)
    if title:
        ax.set_title(title, pad=14, fontweight="normal")

    # Thin white cell separators keep the grid readable without adding visual noise.
    ax.set_xticks(np.arange(-0.5, number_of_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, number_of_classes, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=4.5, width=1.1, pad=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    threshold = maximum * 0.52
    for row_index in range(number_of_classes):
        for column_index in range(number_of_classes):
            text_color = "white" if counts[row_index, column_index] >= threshold else "#1F2933"
            ax.text(
                column_index,
                row_index,
                format_annotation(counts[row_index, column_index], percentages[row_index, column_index]),
                ha="center",
                va="center",
                color=text_color,
                fontsize=annotation_size,
                # Keep the count and percentage grouped, while the parentheses make
                # the two different quantities immediately distinguishable.
                linespacing=1.0,
                fontweight="normal",
            )

    # A restrained outline makes the diagonal immediately visible as the correct class.
    for index in range(number_of_classes):
        ax.add_patch(
            Rectangle(
                (index - 0.5, index - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#123B5D",
                linewidth=1.25,
                zorder=3,
            )
        )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Number of samples", rotation=90, labelpad=12, fontsize=axis_size)
    colorbar.ax.tick_params(labelsize=max(8.5, tick_size - 1), length=3.5, width=0.9)
    colorbar.outline.set_linewidth(0.8)

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def run(
    input_csv: str | Path,
    true_column: str,
    predicted_column: str,
    output_prefix: str | Path | None = None,
    labels: list[str] | None = None,
    *,
    font_family: str = "DejaVu Sans",
    title: str | None = None,
) -> tuple[Path, Path]:
    """Read the CSV and render both publication-ready output formats."""
    input_csv = Path(input_csv).expanduser()
    rows = read_label_rows(
        input_csv,
        true_column=true_column,
        predicted_column=predicted_column,
        require_matching_label_sets=not labels,
    )
    class_labels = labels or infer_labels(rows)
    counts, percentages = build_confusion_matrix(rows, class_labels)
    output_prefix = output_prefix or input_csv.with_name(f"{input_csv.stem}_confusion_matrix")
    return make_plot(
        counts,
        percentages,
        class_labels,
        output_prefix,
        title=title,
        font_family=font_family,
    )


def main(
    input_csv: str | Path | None = None,
    output_prefix: str | Path | None = None,
    true_column: str | None = None,
    predicted_column: str | None = None,
    labels: list[str] | None = None,
    *,
    font_family: str = "DejaVu Sans",
    title: str | None = None,
) -> int:
    """Run interactively when values are omitted, or non-interactively otherwise."""
    try:
        if input_csv is None and true_column is None and predicted_column is None:
            input_csv, true_column, predicted_column = prompt_user_configuration()
        else:
            if input_csv is None:
                entered_filename = input("CSV filename (path or filename in this folder): ").strip()
                if not entered_filename:
                    raise ValueError("No CSV filename was provided.")
                input_csv = Path(entered_filename).expanduser()
            if true_column is None:
                entered_true_column = input(
                    f"True-label column name [{DEFAULT_TRUE_COLUMN}]: "
                ).strip()
                true_column = entered_true_column or DEFAULT_TRUE_COLUMN
            if predicted_column is None:
                entered_predicted_column = input(
                    f"Predicted-label column name [{DEFAULT_PREDICTED_COLUMN}]: "
                ).strip()
                predicted_column = entered_predicted_column or DEFAULT_PREDICTED_COLUMN

        png_path, pdf_path = run(
            input_csv,
            true_column,
            predicted_column,
            output_prefix,
            labels,
            font_family=font_family,
            title=title,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"PNG saved to: {png_path}")
    print(f"PDF saved to: {pdf_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a publication-style n-class confusion matrix from a CSV file."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        help="CSV containing true and predicted labels; if omitted, the script asks for it",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Output path without extension; defaults to <input>_confusion_matrix",
    )
    parser.add_argument(
        "--true-column",
        default=None,
        help="True-label column name; if omitted, the script asks for it",
    )
    parser.add_argument(
        "--predicted-column",
        default=None,
        help="Predicted-label column name; if omitted, the script asks for it",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Explicit class order, useful when a class has zero predicted samples",
    )
    parser.add_argument("--font-family", default="DejaVu Sans", help="Matplotlib font family")
    parser.add_argument("--title", help="Optional figure title")
    return parser.parse_args()


def cli_main() -> int:
    """Parse command-line options and delegate to the interactive-friendly main."""
    args = _parse_args()
    return main(
        input_csv=args.input_csv,
        output_prefix=args.output_prefix,
        true_column=args.true_column,
        predicted_column=args.predicted_column,
        labels=args.labels,
        font_family=args.font_family,
        title=args.title,
    )


if __name__ == "__main__":
    raise SystemExit(cli_main())
