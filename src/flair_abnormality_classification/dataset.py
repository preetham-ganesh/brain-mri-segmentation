import os

import pandas as pd

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

    def load_dataset_file_paths(self) -> None:
        """Loads file paths of images & masks in the dataset.

        Loads file paths of images & masks in the dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Creates absolute directory paths for the following image directories.
        no_abnormality_images_directory_path = check_directory_path_existence(
            "data/processed_data/lgg_mri_segmentation/v{}/no_abnormality/images".format(
                self.model_configuration["dataset"]["version"],
            )
        )
        abnormality_images_directory_path = check_directory_path_existence(
            "data/processed_data/lgg_mri_segmentation/v{}/abnormality/images".format(
                self.model_configuration["dataset"]["version"],
            )
        )

        # Lists names of files in the directories.
        abnormality_image_names = os.listdir(abnormality_images_directory_path)
        no_abnormality_image_names = os.listdir(abnormality_images_directory_path)

        # Iterates across image names in the abnormality class.
        self.file_paths = list()
        for image_name in abnormality_image_names:

            # If first letter of image name is '.', then ignores the image.
            if image_name[0] != ".":
                continue

            # Appends image file path & class 1 to the list.
            self.file_paths.append(
                {
                    "image_file_path": "{}/{}".format(
                        abnormality_images_directory_path, image_name
                    ),
                    "class": 1,
                }
            )

        # Iterates across image names in the abnormality class.
        self.file_paths = list()
        for image_name in no_abnormality_image_names:

            # If first letter of image name is '.', then ignores the image.
            if image_name[0] != ".":
                continue

            # Appends image file path & class 1 to the list.
            self.file_paths.append(
                {
                    "image_file_path": "{}/{}".format(
                        no_abnormality_images_directory_path, image_name
                    ),
                    "class": 0,
                }
            )

        # Converts list of file paths as records into dataframe.
        self.file_paths = pd.DataFrame.from_records(self.file_paths)
        print(
            "No. of valid images in the processed dataset: {}".format(
                len(self.file_paths)
            )
        )
        print(
            "No. of abnormality class images in the processed dataset: {}".format(
                len(self.file_paths["class"].value_counts().get(1, 0))
            )
        )
        print(
            "No. of no abnormality class images in the processed dataset: {}".format(
                len(self.file_paths["class"].value_counts().get(0, 0))
            )
        )
