from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns


def load_heatmap_csv(csv_path):
    """Load a labeled heatmap matrix from a CSV file.

    The first CSV column is used as row labels and the first CSV row's
    remaining fields are used as column labels by pandas.
    """
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    try:
        frame = pd.read_csv(path, index_col=0)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse CSV file: {path}") from exc

    if frame.empty or frame.shape[1] == 0:
        raise ValueError(
            "The CSV must contain a first label column and at least one data column."
        )

    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    # Blank cells and text cells are both invalid heatmap values.
    invalid_cells = numeric_frame.isna()
    if invalid_cells.any().any():
        invalid_locations = [
            f"{row}, {column}"
            for row, column in invalid_cells.stack()[invalid_cells.stack()].index
        ]
        location_text = ", ".join(invalid_locations[:3])
        if len(invalid_locations) > 3:
            location_text += ", ..."
        detail = f" at {location_text}" if location_text else ""
        raise ValueError(f"All heatmap cells must be numeric{detail}.")

    return numeric_frame


def derive_output_path(csv_path):
    """Return the output PNG path beside the input CSV."""
    path = Path(csv_path).expanduser()
    return path.with_name(f"{path.stem}_heatmap.png")


def _diverging_colormap():
    """Muted blue-white-red palette for signed data centered at zero."""
    return LinearSegmentedColormap.from_list(
        "cns_muted_blue_white_red",
        ["#4169A1", "#F7F7F7", "#C84B4B"],
        N=256,
    )


def _sequential_colormap():
    """Muted light-to-dark red palette for data that does not cross zero."""
    return LinearSegmentedColormap.from_list(
        "cns_muted_sequential_red",
        ["#FBF3F1", "#E8AFA6", "#C84B4B", "#7F1D1D"],
        N=256,
    )


def _color_limits(values):
    """Choose sensible color limits, centering signed data at zero."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("The heatmap contains no finite numeric values.")

    data_min = float(finite_values.min())
    data_max = float(finite_values.max())
    if data_min < 0 < data_max:
        limit = max(abs(data_min), abs(data_max))
        return -limit, limit, 0

    if data_min == data_max:
        padding = max(abs(data_min) * 0.05, 1.0)
        return data_min - padding, data_max + padding, None

    return data_min, data_max, None


def create_heatmap(frame, row_title, column_title, output_path, show=True, value_title="Value"):
    """Create and save the heatmap, returning the generated PNG path.

    Signed data (negative and positive values) uses a diverging palette centered
    at zero; all other data uses a sequential palette, so color intensity simply
    tracks magnitude.
    """
    values = frame.to_numpy(dtype=float)
    vmin, vmax, center = _color_limits(values)
    colormap = _diverging_colormap() if center is not None else _sequential_colormap()

    row_count, column_count = frame.shape
    figure_width = max(10, column_count * 0.75 + 3.5)
    figure_height = max(7, row_count * 0.5 + 2.5)

    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    heatmap_options = {
        "data": frame,
        "annot": False,
        "cmap": colormap,
        "vmin": vmin,
        "vmax": vmax,
        "square": True,
        "linewidths": 0.6,
        "linecolor": "white",
        "cbar_kws": {
            "label": value_title,
            "shrink": 0.9,
            "pad": 0.04,
            "aspect": 28,
        },
        "ax": ax,
    }
    if center is not None:
        heatmap_options["center"] = center

    sns.heatmap(**heatmap_options)

    ax.set_xlabel(column_title, fontsize=17, fontweight="normal", labelpad=14)
    ax.set_ylabel(row_title, fontsize=17, fontweight="normal", labelpad=14)
    ax.tick_params(axis="x", labelsize=13, pad=5)
    ax.tick_params(axis="y", labelsize=13, pad=5)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=12)
    colorbar.set_label(value_title, fontsize=14, fontweight="normal", labelpad=12)

    fig.tight_layout()
    output = Path(output_path).expanduser()
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    backend = plt.get_backend().lower()
    if show and "agg" not in backend and backend not in {"pdf", "pgf", "ps", "svg"}:
        plt.show()
    plt.close(fig)
    return output


def main(csv_path=None, row_title=None, column_title=None, show=True, value_title="Value"):
    """Run the CSV-to-heatmap workflow interactively or from a notebook."""
    if csv_path is None:
        csv_input = input("Enter CSV filename or path: ").strip()
    else:
        csv_input = str(csv_path).strip()
    if not csv_input:
        print("Error: CSV filename cannot be empty.")
        return 1

    if row_title is None:
        row_title = input("Enter row title: ").strip()
    else:
        row_title = str(row_title).strip()
    if column_title is None:
        column_title = input("Enter column title: ").strip()
    else:
        column_title = str(column_title).strip()

    try:
        frame = load_heatmap_csv(csv_input)
        output_path = derive_output_path(csv_input)
        create_heatmap(
            frame, row_title, column_title, output_path, show=show, value_title=value_title
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Heatmap saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
