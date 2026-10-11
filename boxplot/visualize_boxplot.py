from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PALETTE = ["#4169A1", "#6A9BBE", "#C84B4B", "#D88C8C"]
EDGE_COLOR = "#243447"


def load_boxplot_csv(csv_path):
    """Load a non-empty CSV file as a DataFrame."""
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse CSV file: {path}") from exc

    if frame.empty:
        raise ValueError("The CSV contains no observations.")
    if len(frame.columns) < 2:
        raise ValueError("The CSV must contain at least two columns.")
    return frame


def _prompt_for_column(frame, role, exclude=None, input_func=input, output_func=print):
    """Prompt the user to choose one column by number or exact name."""
    excluded = set(exclude or [])
    available = [column for column in frame.columns if column not in excluded]
    if not available:
        raise ValueError(f"No available columns remain for the {role} column.")

    output_func(f"\nAvailable columns for the {role} column:")
    for index, column in enumerate(available, start=1):
        output_func(f"  {index}. {column}")

    while True:
        selection = input_func(
            f"Select the {role} column by number or exact name: "
        ).strip()
        if selection.isdigit():
            index = int(selection)
            if 1 <= index <= len(available):
                return available[index - 1]
        elif selection in available:
            return selection

        output_func(
            "Invalid selection. Enter one of the displayed numbers "
            "or an exact column name."
        )


def choose_columns(
    frame,
    category_column=None,
    value_column=None,
    input_func=input,
    output_func=print,
):
    """Choose the categorical label column and numeric y-value column.

    If either column name is omitted, the available columns are displayed and
    the user is prompted to select one by number or exact name.
    """
    columns = list(frame.columns)
    if category_column is None:
        category_column = _prompt_for_column(
            frame, "label/category", input_func=input_func, output_func=output_func
        )
    elif category_column not in columns:
        raise ValueError(f"Label/category column not found: {category_column}")

    if value_column is None:
        value_column = _prompt_for_column(
            frame,
            "y-value",
            exclude={category_column},
            input_func=input_func,
            output_func=output_func,
        )
    elif value_column not in columns:
        raise ValueError(f"Y-value column not found: {value_column}")

    if category_column == value_column:
        raise ValueError("The label/category and y-value columns must be different.")

    return category_column, value_column


def choose_axis_titles(
    category_column,
    value_column,
    category_title=None,
    value_title=None,
    input_func=input,
    output_func=print,
):
    """Ask the user for axis titles when they are not provided."""
    if category_title is None:
        category_title = input_func(
            f"Enter the x-axis title [{category_column}]: "
        ).strip()
    if value_title is None:
        value_title = input_func(
            f"Enter the y-axis title [{value_column}]: "
        ).strip()

    category_title = category_title or str(category_column)
    value_title = value_title or str(value_column)
    output_func(f"X-axis title: {category_title}")
    output_func(f"Y-axis title: {value_title}")
    return category_title, value_title


def validate_boxplot_data(frame, category_column, value_column):
    """Validate selected columns and return a plotting-ready copy."""
    if category_column not in frame.columns:
        raise ValueError(f"Label/category column not found: {category_column}")
    if value_column not in frame.columns:
        raise ValueError(f"Y-value column not found: {value_column}")
    if category_column == value_column:
        raise ValueError("The label/category and y-value columns must be different.")

    category_values = frame[category_column]
    blank_categories = category_values.isna() | category_values.astype(str).str.strip().eq("")
    if blank_categories.any():
        rows = ", ".join(str(index + 2) for index in frame.index[blank_categories][:3])
        raise ValueError(f"The label/category column contains blank values near CSV row(s): {rows}.")

    numeric_values = pd.to_numeric(frame[value_column], errors="coerce")
    invalid_values = numeric_values.isna()
    if invalid_values.any():
        rows = ", ".join(str(index + 2) for index in frame.index[invalid_values][:3])
        raise ValueError(
            f"The y-value column must contain numeric values; "
            f"invalid CSV row(s): {rows}."
        )

    plotting_frame = frame.copy()
    plotting_frame[category_column] = category_values.astype(str).str.strip()
    plotting_frame[value_column] = numeric_values
    return plotting_frame


def derive_output_path(csv_path):
    """Return the output PNG path for the version with observations."""
    return derive_output_paths(csv_path)[0]


def derive_output_paths(csv_path):
    """Return output PNG paths with and without jittered observations."""
    path = Path(csv_path).expanduser()
    with_points = path.with_name(f"{path.stem}_boxplot_with_points.png")
    without_points = path.with_name(f"{path.stem}_boxplot_without_points.png")
    return with_points, without_points


def _ordered_groups(frame, category_column, group_order=None):
    """Return an explicit group order or the order observed in the CSV."""
    observed = list(frame[category_column].drop_duplicates())
    if group_order is None:
        return observed

    missing = [group for group in group_order if group not in observed]
    if missing:
        missing_text = ", ".join(map(str, missing))
        raise ValueError(f"The requested group_order contains unknown group(s): {missing_text}")

    ordered = list(group_order)
    ordered.extend(group for group in observed if group not in ordered)
    return ordered


def create_boxplot(
    frame,
    category_column,
    value_column,
    output_path,
    category_title=None,
    value_title=None,
    show=True,
    title=None,
    group_order=None,
    show_points=True,
):
    """Create and save a generic publication-style boxplot."""
    frame = validate_boxplot_data(frame, category_column, value_column)
    groups = _ordered_groups(frame, category_column, group_order=group_order)
    colors = (PALETTE * ((len(groups) // len(PALETTE)) + 1))[: len(groups)]
    palette = dict(zip(groups, colors))
    category_title = category_title or str(category_column)
    value_title = value_title or str(value_column)

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
    fig, ax = plt.subplots(figsize=(10, 5.8))

    sns.boxplot(
        data=frame,
        x=category_column,
        y=value_column,
        order=groups,
        hue=category_column,
        palette=palette,
        legend=False,
        width=0.62,
        linewidth=1.6,
        # With observations overlaid, outliers are already visible as points.
        # Without them, draw outliers (beyond 1.5 x IQR) so they are not hidden.
        showfliers=not show_points,
        flierprops={
            "marker": "o",
            "markersize": 6,
            "markerfacecolor": "white",
            "markeredgecolor": EDGE_COLOR,
            "markeredgewidth": 1.2,
        },
        boxprops={"edgecolor": EDGE_COLOR, "linewidth": 1.6},
        whiskerprops={"color": EDGE_COLOR, "linewidth": 1.4},
        capprops={"color": EDGE_COLOR, "linewidth": 1.4},
        medianprops={"color": EDGE_COLOR, "linewidth": 2.2},
        ax=ax,
    )

    if show_points:
        sns.stripplot(
            data=frame,
            x=category_column,
            y=value_column,
            order=groups,
            hue=category_column,
            palette=palette,
            legend=False,
            jitter=0.16,
            dodge=False,
            size=5,
            alpha=0.62,
            edgecolor="white",
            linewidth=0.45,
            ax=ax,
        )

    if title:
        ax.set_title(title, fontsize=18, fontweight="bold", pad=16)
    ax.set_xlabel(category_title, fontsize=15, fontweight="normal", labelpad=12)
    ax.set_ylabel(value_title, fontsize=15, fontweight="normal", labelpad=12)
    ax.tick_params(axis="both", labelsize=15, length=5, width=1.3, pad=5)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    ax.yaxis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    fig.subplots_adjust(left=0.22, bottom=0.30, right=0.95, top=0.90)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")

    backend = plt.get_backend().lower()
    if show and "agg" not in backend and backend not in {"pdf", "pgf", "ps", "svg"}:
        plt.show()
    plt.close(fig)
    return output


def main(
    csv_path=None,
    category_column=None,
    value_column=None,
    category_title=None,
    value_title=None,
    output_path=None,
    title=None,
    group_order=None,
    show_points=True,
    show=True,
):
    """Run the generic CSV-to-boxplot workflow interactively or from a notebook."""
    if csv_path is None:
        csv_path = input("Enter the CSV filename or path: ").strip()
    if not csv_path:
        print("Error: no CSV filename was provided.")
        return 1

    try:
        frame = load_boxplot_csv(csv_path)
        category_column, value_column = choose_columns(
            frame,
            category_column=category_column,
            value_column=value_column,
        )
        print(f"Selected label/category column: {category_column}")
        print(f"Selected y-value column: {value_column}")
        category_title, value_title = choose_axis_titles(
            category_column,
            value_column,
            category_title=category_title,
            value_title=value_title,
        )
        if output_path is None:
            output_with_points, output_without_points = derive_output_paths(csv_path)
        else:
            output_with_points = Path(output_path).expanduser()
            output_without_points = output_with_points.with_name(
                f"{output_with_points.stem}_without_points{output_with_points.suffix}"
            )

        create_boxplot(
            frame=frame,
            category_column=category_column,
            value_column=value_column,
            category_title=category_title,
            value_title=value_title,
            output_path=output_with_points,
            title=title,
            group_order=group_order,
            show_points=True,
            show=show,
        )
        create_boxplot(
            frame=frame,
            category_column=category_column,
            value_column=value_column,
            category_title=category_title,
            value_title=value_title,
            output_path=output_without_points,
            title=title,
            group_order=group_order,
            show_points=False,
            show=show,
        )
    except (EOFError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Boxplot with jittered observations saved to: {output_with_points}")
    print(f"Boxplot without observations saved to: {output_without_points}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
