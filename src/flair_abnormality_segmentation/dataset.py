import os

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
