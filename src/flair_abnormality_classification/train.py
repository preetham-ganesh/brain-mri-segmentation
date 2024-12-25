import os

from src.utils import load_json_file
from src.flair_abnormality_classification.dataset import Dataset


class Train(object):
    """Trains the segmentation model based on the configuration."""

    def __init__(self, model_version: str) -> None:
        """Creates object attributes for the Train class.

        Creates object attributes for the Train class.

        Args:
            model_version: A string for the version of the current model.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(model_version, str), "Variable model_version of type 'str'."

        # Initalizes class variables.
        self.model_version = model_version
        self.best_validation_loss = None

    def load_model_configuration(self) -> None:
        """Loads the model configuration file for model version.

        Loads the model configuration file for model version.

        Args:
            None.

        Returns:
            None.
        """
        self.home_directory_path = os.getcwd()
        model_configuration_directory_path = os.path.join(
            self.home_directory_path, "configs/flair_abnormality_classification"
        )
        self.model_configuration = load_json_file(
            f"v{self.model_version}", model_configuration_directory_path
        )

    def load_dataset(self) -> None:
        """Loads the dataset based on model configuration.

        Loads the dataset based on model configuration.

        Args:
            None.

        Returns:
            None.
        """
        # Initializes object for the Dataset class.
        self.dataset = Dataset(self.model_configuration)

        # Loads the metadata for the images in the dataset.
        self.dataset.load_images_metadata

        # Splits file paths into new train, validation & test file paths.
        self.dataset.split_dataset()

        # Converts split data into tensor dataset & slices them based on batch size.
        self.dataset.shuffle_slice_dataset()
