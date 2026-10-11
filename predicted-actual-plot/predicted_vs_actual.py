#!/usr/bin/env python3
"""Create a predicted-vs-actual regression evaluation plot.

The script asks for an input CSV/Excel file and the names of the Actual and
Predicted columns. It then saves a scatter plot with a y=x identity line, a
fitted regression line, and R²/RMSE/MAE metrics.

Usage:
    python predicted_vs_actual.py
    python predicted_vs_actual.py path/to/data.csv
    python predicted_vs_actual.py path/to/data.xlsx --output result.png

Dependencies:
    matplotlib
    openpyxl (only required for .xlsx/.xlsm files)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any


class InputDataError(ValueError):
    """Raised when the input table cannot be used for the plot."""


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _clean_path(value: str) -> Path:
    """Convert terminal input into a usable path, including quoted paths."""
    return Path(value.strip().strip('"').strip("'")).expanduser()


def _validate_headers(headers: list[Any]) -> list[str]:
    cleaned = ["" if _is_blank(header) else str(header).strip() for header in headers]
    if not cleaned or any(not header for header in cleaned):
        raise InputDataError("The first row must contain a name for every column.")

    normalized = [header.casefold() for header in cleaned]
    if len(set(normalized)) != len(normalized):
        raise InputDataError(
            "Column names must be unique (ignoring upper/lower case and leading/trailing spaces)."
        )
    return cleaned


def _read_csv_file(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                headers = _validate_headers(next(reader))
            except StopIteration:
                raise InputDataError("The CSV file is empty.") from None

            rows: list[dict[str, Any]] = []
            for row_number, values in enumerate(reader, start=2):
                if not values or all(_is_blank(value) for value in values):
                    continue
                if len(values) > len(headers):
                    raise InputDataError(
                        f"CSV row {row_number} has {len(values)} values but only {len(headers)} columns."
                    )
                padded = values + [None] * (len(headers) - len(values))
                rows.append(dict(zip(headers, padded)))
    except UnicodeDecodeError as exc:
        raise InputDataError("Could not decode the CSV file as UTF-8.") from exc
    except OSError as exc:
        raise InputDataError(f"Could not read {path}: {exc}") from exc

    if not rows:
        raise InputDataError("The input file contains headers but no data rows.")
    return headers, rows


def _read_excel_file(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError(
            "Reading Excel files requires openpyxl. Install it with "
            "`python -m pip install openpyxl`."
        ) from exc

    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        raise InputDataError(f"Could not read Excel workbook {path}: {exc}") from exc

    try:
        worksheet = workbook.active
        rows = [list(row) for row in worksheet.iter_rows(values_only=True)]
    finally:
        workbook.close()

    while rows and all(_is_blank(value) for value in rows[-1]):
        rows.pop()
    if not rows:
        raise InputDataError("The first worksheet is empty.")

    # Excel often reports formatted-but-empty columns on the right; drop columns
    # that are blank in every row so they are not treated as unnamed headers.
    width = max(len(row) for row in rows)
    rows = [list(row) + [None] * (width - len(row)) for row in rows]
    while width and all(_is_blank(row[width - 1]) for row in rows):
        width -= 1
    rows = [row[:width] for row in rows]

    headers = _validate_headers(rows[0])
    records: list[dict[str, Any]] = []
    for values in rows[1:]:
        if all(_is_blank(value) for value in values):
            continue
        padded = values + [None] * (len(headers) - len(values))
        if len(values) > len(headers):
            raise InputDataError("A worksheet row contains more values than the header row.")
        records.append(dict(zip(headers, padded)))

    if not records:
        raise InputDataError("The first worksheet contains headers but no data rows.")
    return headers, records


def read_table(path: str | Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Read the first table from a CSV or the active worksheet from Excel."""
    input_path = Path(path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if not input_path.is_file():
        raise InputDataError(f"Input path is not a file: {input_path}")

    suffix = input_path.suffix.casefold()
    if suffix == ".csv":
        return _read_csv_file(input_path)
    if suffix in {".xlsx", ".xlsm"}:
        return _read_excel_file(input_path)
    raise InputDataError("Supported input formats are .csv, .xlsx, and .xlsm.")


def resolve_column(headers: list[str], entered: str) -> str:
    """Resolve a column name case-insensitively while preserving its real name."""
    candidate = entered.strip().casefold()
    matches = [header for header in headers if header.casefold() == candidate]
    if not matches:
        raise InputDataError(
            f"Column {entered!r} was not found. Available columns: {', '.join(headers)}."
        )
    return matches[0]


def _prompt_column(headers: list[str], label: str) -> str:
    while True:
        try:
            entered = input(f"{label} column name: ").strip()
        except EOFError as exc:
            raise InputDataError("No column name was provided.") from exc
        try:
            return resolve_column(headers, entered)
        except InputDataError as exc:
            print(f"Error: {exc}")
            print("Please choose one of the listed column names.")


def _to_float(value: Any, column: str, row_number: int) -> float | None:
    if _is_blank(value):
        return None
    if isinstance(value, bool):
        raise InputDataError(f"Column {column!r} has a non-numeric value at data row {row_number}.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise InputDataError(
            f"Column {column!r} has a non-numeric value at data row {row_number}: {value!r}."
        ) from None
    if not math.isfinite(number):
        raise InputDataError(
            f"Column {column!r} has a non-finite value at data row {row_number}: {value!r}."
        )
    return number


def extract_numeric_pairs(
    rows: list[dict[str, Any]], actual_column: str, predicted_column: str
) -> tuple[list[float], list[float], int]:
    """Extract paired numeric values; blank pairs are skipped."""
    available = list(rows[0]) if rows else []
    for column in (actual_column, predicted_column):
        if column not in available:
            raise InputDataError(
                f"Column {column!r} was not found. Available columns: {', '.join(available)}. "
                "Use resolve_column() for case-insensitive matching."
            )
    actual_values: list[float] = []
    predicted_values: list[float] = []
    skipped_rows = 0
    for row_number, row in enumerate(rows, start=2):
        actual = _to_float(row.get(actual_column), actual_column, row_number)
        predicted = _to_float(row.get(predicted_column), predicted_column, row_number)
        if actual is None or predicted is None:
            skipped_rows += 1
            continue
        actual_values.append(actual)
        predicted_values.append(predicted)

    if len(actual_values) < 2:
        raise InputDataError("At least two complete numeric Actual/Predicted pairs are required.")
    return actual_values, predicted_values, skipped_rows


def calculate_metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    mean_actual = sum(actual) / len(actual)
    squared_errors = [(prediction - observed) ** 2 for observed, prediction in zip(actual, predicted)]
    absolute_errors = [abs(prediction - observed) for observed, prediction in zip(actual, predicted)]
    total_sum_of_squares = sum((observed - mean_actual) ** 2 for observed in actual)
    if total_sum_of_squares == 0:
        raise InputDataError("R² cannot be calculated because all Actual values are identical.")

    return {
        "r2": 1.0 - sum(squared_errors) / total_sum_of_squares,
        "rmse": math.sqrt(sum(squared_errors) / len(actual)),
        "mae": sum(absolute_errors) / len(actual),
    }


def _fit_line(actual: list[float], predicted: list[float]) -> tuple[float, float]:
    mean_actual = sum(actual) / len(actual)
    mean_predicted = sum(predicted) / len(predicted)
    denominator = sum((value - mean_actual) ** 2 for value in actual)
    if denominator == 0:
        raise InputDataError("A fitted line cannot be calculated because all Actual values are identical.")
    slope = sum(
        (observed - mean_actual) * (prediction - mean_predicted)
        for observed, prediction in zip(actual, predicted)
    ) / denominator
    intercept = mean_predicted - slope * mean_actual
    return slope, intercept


def make_plot(
    actual: list[float],
    predicted: list[float],
    metrics: dict[str, float],
    actual_column: str,
    predicted_column: str,
    output_path: str | Path,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it with `python -m pip install matplotlib`."
        ) from exc

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    slope, intercept = _fit_line(actual, predicted)
    lower = min(min(actual), min(predicted))
    upper = max(max(actual), max(predicted))
    span = max(upper - lower, 1.0)
    padding = span * 0.06
    axis_min = lower - padding
    axis_max = upper + padding

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.labelweight": "normal",
        "axes.linewidth": 1.5,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    })
    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    ax.scatter(
        actual,
        predicted,
        s=42,
        color="#2F80ED",
        edgecolors="white",
        linewidths=0.7,
        alpha=0.88,
        label="Samples",
        zorder=3,
    )
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        color="#6B7280",
        linestyle="--",
        linewidth=1.5,
        label="y = x reference line",
        zorder=1,
    )
    ax.plot(
        [axis_min, axis_max],
        [slope * axis_min + intercept, slope * axis_max + intercept],
        color="#D95F02",
        linewidth=2.0,
        label="Fitted line",
        zorder=2,
    )
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Actual values", fontweight="normal")
    ax.set_ylabel("Predicted values", fontweight="normal")
    ax.set_title("Predicted vs. Actual")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)
    ax.grid(False)
    ax.tick_params(axis="both", length=5, width=1.3)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.97,
        0.04,
        f"R² = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.3f}\nMAE = {metrics['mae']:.3f}\nn = {len(actual)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9, "pad": 7},
    )
    fig.tight_layout()
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def run(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    path = Path(input_path).expanduser()
    headers, rows = read_table(path)
    print("\nAvailable columns:")
    for index, header in enumerate(headers, start=1):
        print(f"  {index}. {header}")

    actual_column = _prompt_column(headers, "Actual")
    predicted_column = _prompt_column(headers, "Predicted")
    actual, predicted, skipped_rows = extract_numeric_pairs(rows, actual_column, predicted_column)
    metrics = calculate_metrics(actual, predicted)
    output = Path(output_path).expanduser() if output_path else path.with_name(
        f"{path.stem}_predicted_vs_actual.png"
    )
    make_plot(actual, predicted, metrics, actual_column, predicted_column, output)

    print(f"\nComplete pairs used: {len(actual)}")
    if skipped_rows:
        print(f"Rows skipped because one or both selected values were blank: {skipped_rows}")
    print(f"R²   = {metrics['r2']:.4f}")
    print(f"RMSE = {metrics['rmse']:.4f}")
    print(f"MAE  = {metrics['mae']:.4f}")
    print(f"PNG saved to: {output}")
    return output


def cli_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        help="Input .csv, .xlsx, or .xlsm file; if omitted, the script asks for it",
    )
    parser.add_argument("-o", "--output", type=Path, help="PNG output path")
    args = parser.parse_args()

    try:
        input_path = args.input_file
        if input_path is None:
            entered = input("Input file path (CSV or Excel): ").strip()
            if not entered:
                raise InputDataError("No input file path was provided.")
            input_path = _clean_path(entered)
        run(input_path, args.output)
    except (InputDataError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
