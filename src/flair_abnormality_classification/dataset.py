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
        base_directory_path = "data/processed_data/lgg_mri_segmentation/v{}".format(
            self.model_configuration["dataset"]["version"]
        )
        images_directory_paths = {
            "no_abnormality": check_directory_path_existence(
                "{}/no_abnormality/images".format(base_directory_path)
            ),
            "abnormality": check_directory_path_existence(
                "{}/no_abnormality/images".format(base_directory_path)
            ),
        }

        # Iterates across directories in the dataset.
        self.file_paths = list()
        for label, directory_path in enumerate(images_directory_paths.values()):

            # Lists names of files in the directory.
            image_names = os.listdir(directory_path)

            # Iterates across images in the directory.
            for i_id in range(len(image_names)):

                # If first letter of image name is '.', then ignores the image.
                if image_names[i_id][0] != ".":
                    continue

                # Appends image file path & class 1 to the list.
                self.file_paths.append(
                    {
                        "image_file_path": "{}/{}".format(
                            directory_path, image_names[i_id]
                        ),
                        "class": label,
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
