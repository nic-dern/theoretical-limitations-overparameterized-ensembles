from abc import ABC, abstractmethod
from pathlib import Path
import json


class BaseExperimentParameters:
    """Base class for experiment parameters."""

    def __init__(
        self,
        num_training_samples: int,
        data_dimension: int,
        data_generating_function_name: str,
        noise_level: float,
        kernel: str,
        number_simulations_per_size: int,
    ) -> None:
        """Initialize experiment parameters.

        Args:
            num_training_samples : int
                Number of training samples.
            data_dimension : int
                Dimension of input data.
            data_generating_function_name : str
                Name of data generating function.
            noise_level : float
                Level of noise to add.
            kernel : str
                Kernel function name.
            number_simulations_per_size : int
                Number of simulations per size.
        """
        self.num_training_samples = num_training_samples
        self.data_dimension = data_dimension
        self.data_generating_function_name = data_generating_function_name
        self.noise_level = noise_level
        self.kernel = kernel
        self.number_simulations_per_size = number_simulations_per_size

    def save(self, results_path: Path) -> None:
        """Save parameters to JSON file.

        Args:
            results_path : Path
                Path to save parameters.
        """
        with open(results_path / "experiment_parameters.json", "w") as f:
            json.dump(self.__dict__, f, indent=4)


class SubtermExperimentParameters(BaseExperimentParameters):
    """Parameters for subterm experiments."""

    def __init__(
        self,
        setting_to_change: str,
        start: float,
        end: float,
        step: float,
        num_training_samples: int,
        data_dimension: int,
        data_generating_function_name: str,
        noise_level: float,
        kernel: str,
        number_simulations_per_size: int,
        random_seed: int,
    ):
        """Initialize subterm experiment parameters.

        Args:
            setting_to_change : str
                Setting to vary during experiment.
            start : float
                Starting value for the setting.
            end : float
                Ending value for the setting.
            step : float
                Step size for varying the setting.
            num_training_samples : int
                Number of training samples.
            data_dimension : int
                Dimension of input data.
            data_generating_function_name : str
                Name of data generating function.
            noise_level : float
                Level of noise to add.
            kernel : str
                Kernel function name.
            number_simulations_per_size : int
                Number of simulations per size.
            random_seed : int
                Random seed for reproducibility.
        """
        super().__init__(
            num_training_samples,
            data_dimension,
            data_generating_function_name,
            noise_level,
            kernel,
            number_simulations_per_size,
        )
        self.setting_to_change = setting_to_change
        self.start = start
        self.end = end
        self.step = step
        self.random_seed = random_seed


class RandomFeatureModelExperimentParameters(BaseExperimentParameters):
    """Parameters for random feature model experiments."""

    def __init__(
        self,
        num_features_per_model: int,
        max_num_models: int,
        num_training_samples: int,
        data_dimension: int,
        data_generating_function_name: str,
        noise_level: float,
        kernel: str,
        activation_function: str,
        random_weights_distribution: str,
        case_type: str,
        number_simulations_per_size: int,
    ):
        """Initialize random feature model experiment parameters.

        Args:
            num_features_per_model : int
                Number of features per model.
            max_num_models : int
                Maximum number of models in ensemble.
            num_training_samples : int
                Number of training samples.
            data_dimension : int
                Dimension of input data.
            data_generating_function_name : str
                Name of data generating function.
            noise_level : float
                Level of noise to add.
            kernel : str
                Kernel function name.
            activation_function : str
                Name of activation function.
            random_weights_distribution : str
                Type of random weights distribution.
            case_type : str
                Type of case to run.
            number_simulations_per_size : int
                Number of simulations per size.
        """
        super().__init__(
            num_training_samples,
            data_dimension,
            data_generating_function_name,
            noise_level,
            kernel,
            number_simulations_per_size,
        )
        self.num_features_per_model = num_features_per_model
        self.max_num_models = max_num_models
        self.activation_function = activation_function
        self.random_weights_distribution = random_weights_distribution
        self.case_type = case_type


class Experiment(ABC):
    """Base class for all experiments."""

    def __init__(
        self,
        results_path: Path,
        experiment_parameters: BaseExperimentParameters,
        experiment_number: int,
    ) -> None:
        """Initialize experiment.

        Args:
            results_path : Path
                Path to save results.
            experiment_parameters : BaseExperimentParameters
                Parameters for the experiment.
            experiment_number : int
                Unique identifier for experiment.
        """
        self.experiment_parameters = experiment_parameters
        self.experiment_number = experiment_number

        experiment_dir = Path("results") / self._get_experiment_dir_name()
        experiment_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = experiment_dir

        self.experiment_parameters.save(experiment_dir)

    def run_and_visualize_experiment(self) -> None:
        """Run experiment and visualize results."""
        results = self._run_experiment()
        self._visualize_results(results)

    @abstractmethod
    def _run_experiment(self) -> dict:
        """Run the experiment.

        Returns:
            dict
                Dictionary containing experiment results.
        """
        pass

    @abstractmethod
    def _visualize_results(self, results: dict) -> None:
        """Visualize experiment results.

        Args:
            results : dict
                Dictionary containing experiment results.
        """
        pass
