from typing import Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def to_numpy(value: Union[float, torch.Tensor]) -> Union[float, np.ndarray]:
    """Convert a torch tensor or float to numpy array.

    Args:
        value : Union[float, torch.Tensor]
            Input value to convert.

    Returns:
        Union[float, np.ndarray]
            Converted numpy array or float.
    """
    if isinstance(value, float):
        return value
    return value.cpu().numpy()


def save_figure(
    figure: plt.Figure, filename: Union[str, Path], dpi: int = 600, format: str = "pdf"
) -> None:
    """Save a matplotlib figure to a file with high quality settings.

    Args:
        figure : plt.Figure
            The matplotlib figure to save.
        filename : Union[str, Path]
            Output filename.
        dpi : int, optional
            Resolution in dots per inch, by default 600.
        format : str, optional
            Output format (pdf, png, etc), by default "pdf".
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(filename, dpi=dpi, format=format)
