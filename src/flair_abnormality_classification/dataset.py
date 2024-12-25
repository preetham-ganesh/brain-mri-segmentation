import os

from src.utils import check_directory_path_existence

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
