import matplotlib.pyplot as plt
import torch
from overparameterized_ensembles.utils.constants import (
    SIZE_DEFAULT,
    COLORS,
    LINE_STYLES,
    FIG_SIZE,
    SIZE_LARGE,
)

### General Plotting Functions


def collect_additional_points(X_list, y_list, data_dimension):
    additional_points = []
    if data_dimension == 1:
        for X, y in zip(X_list, y_list):
            additional_points.append(convert_to_points_to_plot2d(X, y))
    elif data_dimension == 2:
        for X, y in zip(X_list, y_list):
            additional_points.append(convert_to_points_to_plot3d(X, y))
    return additional_points


### 2D Plot


def plot2d(
    f,
    f_label="Function",
    input_range=None,
    num_samples=100,
    additional_points=None,
    additional_points_labels=None,
    plot_highest_point=False,
    additional_functions=None,
    additional_functions_labels=None,
    title=None,
    plot_main_function=True,
    alpha=[0.5, 0.5, 0.5, 0.5, 0.5],
):
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

    if input_range is None:
        data_dimension = 1
        input_range = (
            torch.zeros(data_dimension) - 5,
            10 * torch.ones(data_dimension) - 5,
        )

    # Generate a range of x values
    x_range = generate_input_data_grid_2d(input_range, num_samples)

    # Compute the function values
    y_values = f(x_range)

    # Plot the main function if plot_main_function is True
    if plot_main_function:
        ax.plot(x_range.numpy(), y_values.numpy(), label=f_label, color=COLORS[0])

    # Plot additional functions if provided
    if additional_functions is not None and additional_functions_labels is not None:
        for i, func_list in enumerate(additional_functions):
            for func in func_list:
                y_values_additional = func(x_range)
                ax.plot(
                    x_range.numpy(),
                    y_values_additional.numpy(),
                    label=additional_functions_labels[i]
                    if func == func_list[0]
                    else None,
                    color=COLORS[i + (1 if plot_main_function else 0)],
                    linestyle=LINE_STYLES[i],
                    alpha=alpha[i],
                    linewidth=2,
                )

    if plot_highest_point:
        # Find the highest point
        max_index = torch.argmax(y_values)
        max_x = x_range[max_index].item()
        max_y = y_values[max_index].item()

        # Mark the highest point
        ax.scatter([max_x], [max_y], color="red", zorder=5)
        ax.annotate(
            "Highest point",
            xy=(max_x, max_y),
            xytext=(max_x, max_y + (max_y * 0.025)),
            # arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=9,
            ha="center",
            color="red",
        )

    # Scatter plot the additional points
    if additional_points is not None and additional_points_labels is not None:
        for i, points in enumerate(additional_points):
            ax.scatter(
                points[:, 0].numpy(),
                points[:, 1].numpy(),
                color=COLORS[
                    i + 1 + (len(additional_functions) if additional_functions else 2)
                ],
                label=additional_points_labels[i],
            )

    # Hide the all but the bottom spines (axis lines); code from https://www.practicaldatascience.org/notebooks/class_5/week_5/46_making_plots_pretty.html
    ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["bottom"].set_bounds(min(x_range), max(x_range))

    if title is not None:
        ax.set_title(title, fontsize=SIZE_LARGE, fontweight="bold")

    ax.legend()

    plt.tight_layout()

    return plt


def generate_input_data_grid_2d(input_range, num_samples: int = 100):
    x_range = torch.linspace(input_range[0][0], input_range[1][0], num_samples)
    # Unsqueeze to make it a 2D tensor
    x_range = x_range.unsqueeze(1)
    return x_range


def convert_to_points_to_plot2d(X, y):
    points_to_plot = [[X[i][0].item(), y[i].item()] for i in range(len(X))]
    points_to_plot = torch.tensor(points_to_plot)
    return points_to_plot


### 3D Plot


def plot3d(
    f,
    f_label="Function",
    input_range=None,
    num_samples=100,
    additional_points=None,
    additional_points_labels=None,
    plot_highest_point=False,
    additional_functions=None,
    additional_functions_labels=None,
    title=None,
):
    if input_range is None:
        data_dimension = 2
        input_range = (
            torch.zeros(data_dimension) - 5,
            10 * torch.ones(data_dimension) - 5,
        )

    # Close all open figures to avoid conflicts
    plt.close("all")
    # Reset to default settings
    plt.rcdefaults()
    plt.rcParams["text.usetex"] = False

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    plt.rcParams["text.usetex"] = True

    plt.rc("font", family="Roboto")  # controls default font
    plt.rc("font", weight="normal")  # controls default font
    plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE_DEFAULT)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE_DEFAULT)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

    # Generate a grid of points over the input range
    x0, x1, X_grid = generate_input_data_grid_3d(input_range, num_samples)

    # Compute the function values
    y_values = f(X_grid).reshape(x0.shape)

    # Plot the surface of the main function
    ax.plot_surface(
        x0.numpy(),
        x1.numpy(),
        y_values.numpy(),
        alpha=0.2,
        color=COLORS[0],
        label=f_label,
    )

    # Plot additional functions if provided
    if additional_functions is not None and additional_functions_labels is not None:
        for i, func in enumerate(additional_functions):
            y_values_additional = func(X_grid).reshape(x0.shape)
            ax.plot_surface(
                x0.numpy(),
                x1.numpy(),
                y_values_additional.numpy(),
                alpha=0.2,
                color=COLORS[i + 1],
                label=additional_functions_labels[i],
            )

    if plot_highest_point:
        # Find the highest point
        max_index = torch.argmax(y_values)
        max_x0 = x0.flatten()[max_index].item()
        max_x1 = x1.flatten()[max_index].item()
        max_y = y_values.flatten()[max_index].item()

        # Mark the highest point
        ax.scatter([max_x0], [max_x1], [max_y], color="red", s=50, zorder=5)
        ax.text(
            max_x0,
            max_x1,
            max_y + (max_y * 0.05),
            "Highest point",
            color="red",
            fontsize=9,
            ha="center",
        )

    # Scatter plot the additional points
    if additional_points is not None and additional_points_labels is not None:
        for i, points in enumerate(additional_points):
            ax.scatter(
                points[:, 0].numpy(),
                points[:, 1].numpy(),
                points[:, 2].numpy(),
                color=COLORS[
                    i + 1 + (len(additional_functions) if additional_functions else 0)
                ],
                label=additional_points_labels[i],
            )

    # Hide the all but the bottom spines (axis lines); code from https://www.practicaldatascience.org/notebooks/class_5/week_5/46_making_plots_pretty.html
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("lower")
    # ax.xaxis.set_ticks_position("bottom")

    # Set bounds for the x-axis
    ax.spines["bottom"].set_bounds(min(x0.flatten()), max(x0.flatten()))
    ax.spines["left"].set_bounds(min(x1.flatten()), max(x1.flatten()))

    if title is not None:
        ax.set_title(title, fontsize=SIZE_LARGE, fontweight="bold")

    ax.legend()

    plt.tight_layout()

    return plt


def generate_input_data_grid_3d(input_range=None, num_samples: int = 100):
    if input_range is None:
        data_dimension = 2
        input_range = (
            torch.zeros(data_dimension) - 5,
            10 * torch.ones(data_dimension) - 5,
        )

    x0_range = torch.linspace(input_range[0][0], input_range[1][0], num_samples)
    x1_range = torch.linspace(input_range[0][1], input_range[1][1], num_samples)
    x0, x1 = torch.meshgrid(x0_range, x1_range, indexing="ij")
    X_grid = torch.stack([x0.flatten(), x1.flatten()], dim=1)

    return x0, x1, X_grid


def convert_to_points_to_plot3d(X, y):
    points_to_plot = [
        [X[i][0].item(), X[i][1].item(), y[i].item()] for i in range(len(X))
    ]
    points_to_plot = torch.tensor(points_to_plot)
    return points_to_plot
