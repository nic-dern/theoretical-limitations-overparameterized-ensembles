import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from overparameterized_ensembles.utils.constants import (
    COLORS,
    LINE_STYLES,
    SIZE_DEFAULT,
    SIZE_LARGE,
    FIG_SIZE,
)


def plot_development_of_mean(
    estimates: torch.Tensor,
    every_xth: int,
    y_lim_min: Optional[float] = -0.10,
    y_lim_max: Optional[float] = 0.10,
) -> plt.Figure:
    """
    Plot the development of the mean of the estimates over time, i.e., for the first t estimates for every time step t.
    This should be done for every dimension of the estimates. The first dimension is the time dimension, the second dimension is the dimension of the estimates.

    Args:
    estimates : torch.Tensor
        A tensor of estimates with shape (time_steps, dimensions).
    every_xth : int
        The step size for calculating the mean.
    y_lim_min : float, optional
        The minimum y-axis limit for the plot.
    y_lim_max : float, optional
        The maximum y-axis limit for the plot.

    Returns:
    plt.Figure
        The matplotlib figure object containing the plot.
    """
    if not isinstance(estimates, torch.Tensor):
        raise TypeError("estimates must be a torch.Tensor")

    # Calculate the mean of the estimates for every time step
    time_steps = range(1, estimates.shape[0], every_xth)

    mean_estimates_per_time = torch.stack(
        [torch.mean(estimates[:t], dim=0) for t in time_steps]
    )

    mean_estimates_per_time = mean_estimates_per_time.squeeze()

    # Plot the development of the mean of the estimates over time
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for i in range(mean_estimates_per_time.shape[1]):
        ax.plot(
            time_steps,
            mean_estimates_per_time[:, i].numpy(),
            label=f"Dimension {i}",
        )

    ax.set_xlabel("Number of Estimates")
    ax.set_ylabel("Mean of Estimates")
    ax.set_ylim(y_lim_min, y_lim_max)
    # ax.legend()

    return fig


def plot_distribution(
    values: torch.Tensor,
    title: str,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    x_label: str = "Sample value",
):
    """
    Plot the distribution of the values.

    Args:
    values : torch.Tensor
        The values to plot.
    title : str
        The title of the plot.
    x_min : float, optional
        The minimum x-axis limit for the plot.
    x_max : float, optional
        The maximum x-axis limit for the plot.
    x_label : str, optional
        The label for the x-axis.

    Returns:
    plt.Figure
        The matplotlib figure object containing the plot.

    Raises:
    TypeError: If values is not a torch.Tensor.
    """
    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor")
    if not isinstance(title, str):
        raise TypeError("title must be a str")

    # Close all open figures to avoid conflicts
    plt.close("all")
    # Reset to default settings
    plt.rcdefaults()
    plt.rcParams["text.usetex"] = False

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    plt.rcParams["text.usetex"] = True

    plt.rc("font", family="Roboto")  # controls default font
    plt.rc("font", weight="normal")  # controls default font
    plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

    # Throw out the upper and lower percent to get a better view of the distribution
    values = values[values < values.quantile(0.999)]
    values = values[values > values.quantile(0.001)]

    ax.hist(values.cpu().numpy(), bins=1000, color=COLORS[0], alpha=0.7)

    # Set the x-axis limits if provided
    if x_min is not None and x_max is not None:
        ax.set_xlim(x_min, x_max)
    else:
        # Convert the tensor to a numpy array
        values_np = values.cpu().numpy()

        # Calculate the 2.5th and 97.5th percentiles
        x_min, x_max = np.percentile(values_np, [0.1, 99.9])

        # Set the x-axis limits
        ax.set_xlim(x_min, x_max)

    # Calculate the sample mean
    sample_mean = values.mean().item()

    # Add a dotted red vertical line at the sample mean
    ax.axvline(
        sample_mean,
        color="#D55E00",
        linestyle="--",
        linewidth=1,
        label=f"Mean: {sample_mean:.2f}",
    )

    # Hide the all but the bottom spines (axis lines); code from https://www.practicaldatascience.org/notebooks/class_5/week_5/46_making_plots_pretty.html
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    ax.set_title(title, fontsize=SIZE_LARGE, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=SIZE_DEFAULT)
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()

    return fig


def plot_graph(
    x_values: List[float],
    y_values: List[List[float]],
    labels: List[str],
    x_label: str,
    y_label: str,
    title: str = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    loglog: bool = False,
    vline_x: Optional[float] = None,
    vline_text: Optional[str] = None,
    decay_slope: Optional[float] = None,
    plot_legend: bool = True,
    linestyles: List[str] = None,
) -> plt:
    """
    This function plots a graph based on the provided parameters.

    Parameters:
        x_values (List[float]): The x-values for the plot.
        y_values (List[List[float]]): The y-values for the plot. Each list in y_values corresponds to a line on the plot.
        labels (List[str]): The labels for the lines. Each label corresponds to a list in y_values.
        y_label (str): The label for the y-axis.
        title (str): The title of the plot.
        xlim (Optional[Tuple[float, float]]): The limits for the x-axis. If None, the limits are determined automatically.
        ylim (Optional[Tuple[float, float]]): The limits for the y-axis. If None, the limits are determined automatically.
        loglog (bool): If True, both x and y axes will be logarithmic.
        vline_x (Optional[float]): The x-coordinate for the vertical line. If None, no vertical line is plotted.
        vline_text (Optional[str]): The text to display at the bottom of the vertical line. If None, no text is displayed.
        decay_slope (Optional[float]): The slope of the decaying line. If None, no decaying line is plotted.
        plot_legend (bool): If True, the legend is displayed.
        linestyles (List[str]): The linestyles for the lines. If None, the default linestyles are used.

    Returns:
        plt: The matplotlib plot object.
    """
    if len(y_values) != len(labels):
        raise ValueError("The length of y_values and labels must be the same.")

    # Close all open figures to avoid conflicts
    plt.close("all")

    # Reset to default settings
    plt.rcdefaults()
    plt.rcParams["text.usetex"] = False

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    plt.rcParams["text.usetex"] = True

    plt.rc("font", family="Roboto")  # controls default font
    plt.rc("font", weight="normal")  # controls default font
    plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE_DEFAULT)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE_DEFAULT)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    for i in range(len(y_values)):
        ax.plot(
            x_values,
            y_values[i],
            label=labels[i],
            color=COLORS[i],
            linewidth=2,
            linestyle=LINE_STYLES[i % len(LINE_STYLES)]
            if linestyles is None
            else linestyles[i],
        )

    # Add diagonal grid lines with the specified slope if decay_slope is given
    if decay_slope is not None:
        x_start = x_values[0]
        y_max = max(
            [max(y) for y in y_values]
        )  # Get the maximum y value across all lines
        y_min = min(
            [min(y) for y in y_values]
        )  # Get the minimum y value across all lines

        # Generate diagonal grid lines at multiple y-start points evenly spaced (in the log space) between y_min and y_max
        y_start_points = np.logspace(np.log10(y_min), np.log10(y_max), num=10)

        for y_start in y_start_points:
            # Calculate the y-values for the grid lines
            diagonal_y_values = [
                y_start * (x / x_start) ** decay_slope for x in x_values
            ]

            # Plot the diagonal lines as grey dotted lines
            ax.plot(
                x_values,
                diagonal_y_values,
                color="grey",
                linestyle="dotted",
                linewidth=1,
            )

    ax.set_xlabel(x_label, fontsize=SIZE_DEFAULT)
    ax.set_ylabel(y_label, fontsize=SIZE_DEFAULT)

    if title is not None:
        ax.set_title(title, fontsize=SIZE_LARGE, fontweight="bold")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if vline_x is not None:
        ax.axvline(x=vline_x, color="#D55E00", linestyle="dotted")
        if vline_text is not None:
            ax.text(
                vline_x * 1.07,
                (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2,
                vline_text,
                color="#D55E00",
                ha="left",
                va="center",
            )

    # Hide the all but the bottom spines (axis lines); code from https://www.practicaldatascience.org/notebooks/class_5/week_5/46_making_plots_pretty.html
    ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["bottom"].set_bounds(min(x_values), max(x_values))

    if plot_legend:
        ax.legend()
    # ax.grid(True)

    plt.tight_layout()

    return plt


def plot_multiple_lines(
    x_values: List[float],
    y_values: List[List[float]],
    x_label: str,
    y_label: str,
    title: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    loglog: bool = False,
) -> plt:
    """
    This function plots multiple lines on the same plot based on the provided parameters.

    Parameters:
        x_values (List[float]): The x-values for the plot.
        y_values (List[List[float]]): The y-values for the plot. Each list in y_values corresponds to a line on the plot.
        y_label (str): The label for the y-axis.
        title (str): The title of the plot.
        xlim (Optional[Tuple[float, float]]): The limits for the x-axis. If None, the limits are determined automatically.
        ylim (Optional[Tuple[float, float]]): The limits for the y-axis. If None, the limits are determined automatically.
        loglog (bool): If True, both x and y axes will be logarithmic.

    Returns:
        plt: The matplotlib plot object.
    """
    # Close all open figures to avoid conflicts
    plt.close("all")

    # Reset to default settings
    plt.rcdefaults()
    plt.rcParams["text.usetex"] = False

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    plt.rcParams["text.usetex"] = True

    plt.rc("font", family="Roboto")  # controls default font
    plt.rc("font", weight="normal")  # controls default font
    plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE_DEFAULT)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE_DEFAULT)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    for i in range(len(y_values)):
        ax.plot(x_values, y_values[i], color=COLORS[0], alpha=0.1)

    ax.set_xlabel(x_label, fontsize=SIZE_DEFAULT)
    ax.set_ylabel(y_label, fontsize=SIZE_DEFAULT)
    ax.set_title(title, fontsize=SIZE_LARGE, fontweight="bold")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Hide the all but the bottom spines (axis lines); code from https://www.practicaldatascience.org/notebooks/class_5/week_5/46_making_plots_pretty.html
    ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["bottom"].set_bounds(min(x_values), max(x_values))

    plt.tight_layout()

    return plt
