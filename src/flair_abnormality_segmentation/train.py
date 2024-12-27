import os

import tensorflow as tf

from src.utils import load_json_file
from src.flair_abnormality_segmentation.dataset import Dataset
from src.flair_abnormality_segmentation.model import UNet


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
            self.home_directory_path, "configs/flair_abnormality_segmentation"
        )
        self.model_configuration = load_json_file(
            f"v{self.model_version}", model_configuration_directory_path
        )

    def load_dataset(self) -> None:
        """Loads file paths of images & masks in the dataset.

        Loads file paths of images & masks in the dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Initializes object for the Dataset class.
        self.dataset = Dataset(self.model_configuration)

        # Loads the metadata for the images in the dataset.
        self.dataset.load_dataset_file_paths()

        # Splits file paths into new train, validation & test file paths.
        self.dataset.split_dataset()

        # Converts split data into tensor dataset & slices them based on batch size.
        self.dataset.shuffle_slice_dataset()

    def load_model(self) -> None:
        """Loads model & other utilies based on model configuration.

        Loads model & other utilies based on model configuration.

        Args:
            None.

        Returns:
            None.
        """
        # Loads model for current model configuration.
        self.model = UNet(self.model_configuration)

        # Loads the optimizer.
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.model_configuration["model"]["learning_rate"]
        )

        # Creates checkpoint manager for the neural network model.
        self.checkpoint_directory_path = os.path.join(
            self.home_directory_path,
            "models/flair_abnormality_segmentation",
            f"v{self.model_version}",
            "checkpoints",
        )
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.checkpoint_directory_path, max_to_keep=1
        )
        print("Finished loading model for current configuration.")
        print()
