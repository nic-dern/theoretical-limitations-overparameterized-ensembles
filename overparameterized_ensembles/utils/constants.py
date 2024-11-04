# Description: Constants used in the project.

from typing import List, Final

# Experiment constants
NUMBER_TEST_SAMPLES: Final[int] = 1000
ZERO_REGULARIZATION: Final[float] = 1e-8  # Used instead of 0 to avoid numerical issues
NUMBER_OF_MODELS_FOR_VARIANCE: Final[int] = 20000  # Set to 4000 fo big experiments
DEFAULT_RANDOM_SEED: Final[int] = 42
DEFAULT_TEST_SAMPLES: Final[int] = 100

# Visualization constants
COLORS: Final[List[str]] = [
    "#56B4E9",  # blue
    "#CC79A7",  # red-purple
    "#009E73",  # green
    "#D55E00",  # red-orange
    "#E69F00",  # orange
]

LINE_STYLES: Final[List[str]] = ["-", "--", "-."]
SIZE_DEFAULT: Final[int] = 12
SIZE_LARGE: Final[int] = 16
FIG_SIZE: Final[tuple[float, float]] = (6.4, 4)

# Computation constants
FITTING_PROCEDURE: Final[str] = "lstsq"  # Alternative: "cholesky"
