import os

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

from typing import Dict, List, Any


class Dataset(object):
    """Loads the dataset based on the model configuration."""

    def __init__(self, model_configuration: Dict[str, Any]) -> None:
        """Creates object attributes for the Dataset class.

        Creates object attributes for the Dataset class.

        Args:
            model_configuration: A dictionary for the configuration of model's current version.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."

        # Initalizes class variables.
        self.model_configuration = model_configuration

    def load_dataset_file_paths(self) -> None:
        """Loads file paths of images & masks in the dataset.

        Loads file paths of images & masks in the dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Creates absolute directory paths for the following image directories.
        base_directory_path = os.path.join(
            "data/processed_data/lgg_mri_segmentation/",
            f"v{self.model_configuration['dataset']['version']}",
        )

        # Lists names of files in the directory.
        image_names = os.listdir(os.path.join(base_directory_path, "images"))

        # Creates empty lists to store file paths of images & masks.
        self.images_file_paths, self.masks_file_paths = list(), list()

        # Iterates across images in the directory.
        for i_id in range(len(image_names)):
            image_file_path = os.path.join(
                base_directory_path, "images", image_names[i_id]
            )
            mask_file_path = os.path.join(
                base_directory_path, "masks", image_names[i_id]
            )

            # If the image & mask files exist, appends their file paths to the respective lists.
            if os.path.isfile(image_file_path) and os.path.isfile(mask_file_path):
                self.images_file_paths.append(image_file_path)
                self.masks_file_paths.append(mask_file_path)
        print(
            f"No. of valid images in the processed dataset: {len(self.images_file_paths)}"
        )
        print()

    def split_dataset(self) -> None:
        """Splits file paths into new train, validation & test file paths.

        Splits file paths into new train, validation & test file paths.

        Args:
            None.

        Returns:
            None.
        """
        # Splits the original images data into new train, validation & test data.
        (
            self.train_images_file_paths,
            self.test_images_file_paths,
            self.train_masks_file_paths,
            self.test_masks_file_paths,
        ) = train_test_split(
            self.images_file_paths,
            self.masks_file_paths,
            test_size=self.model_configuration["dataset"]["split_percentage"]["test"],
            shuffle=True,
            random_state=42,
        )
        (
            self.train_images_file_paths,
            self.validation_images_file_paths,
            self.train_masks_file_paths,
            self.validation_masks_file_paths,
        ) = train_test_split(
            self.train_images_file_paths,
            self.train_masks_file_paths,
            test_size=self.model_configuration["dataset"]["split_percentage"][
                "validation"
            ],
            shuffle=True,
            random_state=42,
        )

        # Stores size of new train, validation and test data.
        self.n_train_examples = len(self.train_images_file_paths)
        self.n_validation_examples = len(self.validation_images_file_paths)
        self.n_test_examples = len(self.test_images_file_paths)

        print(f"No. of examples in the new train data: {self.n_train_examples}")
        print(
            f"No. of examples in the new validation data: {self.n_validation_examples}"
        )
        print(f"No. of examples in the new test data: {self.n_test_examples}")
        print()

    def shuffle_slice_dataset(self) -> None:
        """Converts split data into tensor dataset & slices them based on batch size.

        Converts split data into input & target data. Zips the input & target data, and slices them based on batch size.

        Args:
            None.

        Returns:
            None.
        """
        # Zips images & classes into single tensor, and shuffles it.
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.train_images_file_paths, self.train_masks_file_paths)
        )
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(
            (self.validation_images_file_paths, self.validation_masks_file_paths)
        )
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (self.test_images_file_paths, self.test_masks_file_paths)
        )

        # Slices the combined dataset based on batch size, and drops remainder values.
        self.batch_size = self.model_configuration["model"]["batch_size"]
        self.train_dataset = self.train_dataset.batch(
            self.batch_size, drop_remainder=True
        )
        self.validation_dataset = self.validation_dataset.batch(
            self.batch_size, drop_remainder=True
        )
        self.test_dataset = self.test_dataset.batch(
            self.batch_size, drop_remainder=True
        )

        # Computes number of steps per epoch for all dataset.
        self.n_train_steps_per_epoch = self.n_train_examples // self.batch_size
        self.n_validation_steps_per_epoch = (
            self.n_validation_examples // self.batch_size
        )
        self.n_test_steps_per_epoch = self.n_test_examples // self.batch_size

        print(f"No. of train steps per epoch: {self.n_train_steps_per_epoch}")
        print(f"No. of validation steps per epoch: {self.n_validation_steps_per_epoch}")
        print(f"No. of test steps per epoch: {self.n_test_steps_per_epoch}")
        print()

    def threshold_image(self, image: np.ndarray) -> np.ndarray:
        """Thresholds image to have better distinction of regions in image.

        Thresholds image to have better distinction of regions in image.

        Args:
            image: A NumPy array for the image.

        Returns:
            A NumPy array for the thresholded version of the image.
        """
        # Checks type & values of arguments.
        assert isinstance(
            image, np.ndarray
        ), "Variable image should be of type 'numpy.ndarray'."

        # Thresholds image to have better distinction of regions in image.
        thresholded_image = np.where(
            image > self.model_configuration["model"]["threshold"], 255, 0
        )
        return thresholded_image
