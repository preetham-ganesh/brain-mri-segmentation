import os

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

from src.utils import check_directory_path_existence
from src.preprocess_data import load_image

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

    def load_images_metadata(self) -> None:
        """Loads the metadata for the images in the dataset.

        Loads the metadata for the images in the dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Creates absolute directory paths for the following image directories.
        base_directory_path = (
            "data/processed_data/lgg_mri_segmentation/"
            + f"v{self.model_configuration['dataset']['version']}"
        )
        images_directory_paths = {
            "no_abnormality": check_directory_path_existence(
                os.path.join(base_directory_path, "no_abnormality/images")
            ),
            "abnormality": check_directory_path_existence(
                os.path.join(base_directory_path, "abnormality/images")
            ),
        }

        # Iterates across directories in the dataset.
        self.images_file_paths, self.labels = list(), list()
        for label, directory_path in enumerate(images_directory_paths.values()):

            # Lists names of files in the directory.
            image_names = os.listdir(directory_path)

            # Iterates across images in the directory.
            for i_id in range(len(image_names)):

                # If first letter of image name is '.', then ignores the image.
                if image_names[i_id][0] == ".":
                    continue

                # Appends image file path & class to the list.
                self.images_file_paths.append(
                    os.path.join(directory_path, image_names[i_id])
                )
                self.labels.append(label)
        print(
            f"No. of valid images in the processed dataset: {len(self.images_file_paths)}"
        )
        print()
        print(
            f"No. of abnormality class images in the processed dataset: {self.labels.count(1)}"
        )
        print(
            f"No. of no abnormality class images in the processed dataset: {self.labels.count(0)}"
        )
        print()

    def split_dataset(self) -> None:
        """Splits file paths into new train, validation & test file paths.

        Splits file paths into new train, validation & test file paths in a stratified manner.

        Args:
            None.

        Returns:
            None.
        """
        # Splits the original images data into new train, validation & test data (in stratified manner).
        (
            self.train_images_file_paths,
            self.test_images_file_paths,
            self.train_labels,
            self.test_labels,
        ) = train_test_split(
            self.images_file_paths,
            self.labels,
            test_size=self.model_configuration["dataset"]["split_percentage"]["test"],
            shuffle=True,
            stratify=self.labels,
            random_state=42,
        )
        (
            self.train_images_file_paths,
            self.validation_images_file_paths,
            self.train_labels,
            self.validation_labels,
        ) = train_test_split(
            self.train_images_file_paths,
            self.train_labels,
            test_size=self.model_configuration["dataset"]["split_percentage"][
                "validation"
            ],
            shuffle=True,
            stratify=self.labels,
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
            (self.train_image_file_paths, self.train_labels)
        )
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(
            (self.validation_image_file_paths, self.validation_labels)
        )
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (self.test_image_file_paths, self.test_labels)
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

    def load_input_target_batches(
        self, image_file_paths: List[str], labels: List[int]
    ) -> List[tf.Tensor]:
        """Loads input & target batches for images & labels in current batch.

        Loads input & target batches for images & labels in current batch.

        Args:
            image_file_paths: A list of strings for locations of images in current batch.
            labels: A list of integers for classes the images in current batch belongs to.

        Returns:
            A list of tensors for the input & target batches generated from images & labels.
        """
        # Checks types & values of arguments.
        assert isinstance(
            image_file_paths, list
        ), "Variable image_file_paths should be of type 'list'."
        assert isinstance(labels, list), "Variable labels should be of type 'list'."

        # Reshapes input batch into (batch_size, final_image_height, final_image_width).
        n_images = len(image_file_paths)
        input_batch = np.zeros(
            shape=(
                n_images,
                self.model_configuration["model"]["final_image_height"],
                self.model_configuration["model"]["final_image_width"],
                self.model_configuration["model"]["n_channels"],
            ),
            dtype=np.float32,
        )

        # Iterates across images in the batch.
        for i_id in range(n_images):

            # Loads the image for the current image path.
            input_image = load_image(str(image_file_paths[i_id][0], "UTF-8"))

            # Adds loaded & preprocessed input & target images to corresponding batch arrays.
            input_batch[i_id, :, :, :] = input_image

        # Converts images into tensor, and converts pixels into 0 - 1 range.
        input_batch = tf.convert_to_tensor(input_batch, dtype=tf.float32)
        input_batch /= 255.0

        # Converts labels into categorical tensor.
        target_batch = tf.keras.utils.to_categorical(
            labels, num_classes=self.model_configuration["model"]["n_classes"]
        )
        target_batch = tf.convert_to_tensor(target_batch, dtype=tf.int8)
        return [input_batch, target_batch]
