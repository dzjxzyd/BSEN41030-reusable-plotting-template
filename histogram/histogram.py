"""Interactive multi-column histogram generator."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xlsm"}
DEFAULT_BINS = 40


def load_data(data_path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    path = Path(data_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    try:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        else:
            frame = pd.read_excel(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"The data file is empty: {path}") from exc
    except ImportError as exc:
        raise ValueError(
            "Reading Excel files requires openpyxl. Install it with `python -m pip install openpyxl`."
        ) from exc
    except (pd.errors.ParserError, OSError, ValueError) as exc:
        raise ValueError(f"Could not read data file: {path}") from exc

    if frame.empty or frame.shape[1] == 0:
        raise ValueError("The data file must contain at least one row and one column.")
    return frame


def _finite_numeric_values(series: pd.Series) -> np.ndarray:
    """Convert a column to finite numeric values, dropping invalid entries."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns whose non-blank values are all numeric (at least one finite)."""
    columns = []
    for column in frame.columns:
        series = frame[column]
        non_blank = series.notna() & series.astype(str).str.strip().ne("")
        converted = pd.to_numeric(series[non_blank], errors="coerce")
        if converted.notna().all() and _finite_numeric_values(series).size:
            columns.append(column)
    return columns


def parse_column_selection(selection: str, column_count: int) -> list[int]:
    """Parse one-based comma-separated column numbers into zero-based indices."""
    if not selection or not selection.strip():
        raise ValueError("Column selection cannot be empty.")

    indices: list[int] = []
    for token in selection.split(","):
        token = token.strip()
        try:
            one_based = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid column number: {token!r}") from exc
        if not 1 <= one_based <= column_count:
            raise ValueError(f"Column number must be between 1 and {column_count}.")
        zero_based = one_based - 1
        if zero_based in indices:
            raise ValueError(f"Duplicate column selection: {one_based}")
        indices.append(zero_based)
    return indices


def _validate_bins(bins: int) -> int:
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins <= 0:
        raise ValueError("The number of bins must be a positive integer.")
    return int(bins)


def derive_output_path(data_path: str | Path) -> Path:
    """Return the PNG path beside the input file."""
    path = Path(data_path).expanduser()
    return path.with_name(f"{path.stem}_histogram.png")


def create_histogram(
    frame: pd.DataFrame,
    columns: list[str],
    output_path: str | Path,
    bins: int = DEFAULT_BINS,
    show: bool = True,
) -> Path:
    """Create and save an overlaid histogram for selected DataFrame columns."""
    bins = _validate_bins(bins)
    if not columns:
        raise ValueError("Select at least one column.")

    series_values: list[np.ndarray] = []
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"Column not found: {column}")
        values = _finite_numeric_values(frame[column])
        if values.size == 0:
            raise ValueError(f"Column contains no finite numeric values: {column}")
        series_values.append(values)

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.labelsize": 15,
            "axes.labelweight": "normal",
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.linewidth": 1.5,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(left=0.15, bottom=0.16, right=0.96, top=0.95)
    colors = sns.color_palette("colorblind", n_colors=len(columns))
    # Shared bin edges keep bar widths identical, so overlaid counts are comparable.
    bin_edges = np.histogram_bin_edges(np.concatenate(series_values), bins=bins)

    for column, values, color in zip(columns, series_values, colors):
        ax.hist(
            values,
            bins=bin_edges,
            color=color,
            alpha=0.58,
            edgecolor="#FFFFFF",
            linewidth=0.8,
            label=column,
        )

    ax.set_xlabel("Value", fontweight="normal", labelpad=14)
    ax.set_ylabel("Frequency", fontweight="normal", labelpad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.grid(False)
    ax.tick_params(axis="both", length=5, width=1.3, labelsize=15, pad=5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # Place the legend outside the axes so it never hides bars.
    ax.legend(fontsize=13, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")

    backend = plt.get_backend().lower()
    if show and "agg" not in backend and backend not in {"pdf", "pgf", "ps", "svg"}:
        plt.show()
    plt.close(fig)
    return output


def _prompt_for_file() -> Path:
    response = input("Enter data file path: ").strip()
    if not response:
        raise ValueError("No data file path was provided.")
    return Path(response).expanduser()


def _choose_columns(frame: pd.DataFrame) -> list[str]:
    columns = numeric_columns(frame)
    if not columns:
        raise ValueError("No numeric columns are available for plotting.")

    print("\nAvailable numeric columns:")
    for number, column in enumerate(columns, start=1):
        print(f"  {number}. {column}")
    response = input(
        "Enter column numbers (comma-separated; press Enter to select all): "
    ).strip()
    if not response:
        return columns
    indices = parse_column_selection(response, len(columns))
    return [columns[index] for index in indices]


def _choose_bins() -> int:
    response = input(f"Enter number of histogram bins [{DEFAULT_BINS}]: ").strip()
    if not response:
        return DEFAULT_BINS
    try:
        return _validate_bins(int(response))
    except ValueError as exc:
        raise ValueError("The number of bins must be a positive integer.") from exc


def main(show: bool = True) -> int:
    """Run the interactive file, column, and bin selection workflow."""
    try:
        data_path = _prompt_for_file()
        frame = load_data(data_path)
        columns = _choose_columns(frame)
        bins = _choose_bins()
        output_path = create_histogram(frame, columns, derive_output_path(data_path), bins=bins, show=show)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Histogram saved to: {output_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively create an overlaid histogram.")
    parser.add_argument("--no-show", action="store_true", help="Save the figure without opening a window.")
    args = parser.parse_args()
    raise SystemExit(main(show=not args.no_show))
