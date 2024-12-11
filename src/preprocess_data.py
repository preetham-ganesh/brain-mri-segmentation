import os
import sys
import warnings
import argparse
import zipfile


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")


from src.utils import check_directory_path_existence

from typing import Dict, List


def extract_data_from_zip_file() -> None:
    """Extracts files from downloaded data zip file.

    Extracts files from downloaded data zip file.

    Args:
        None.

    Returns:
        None.
    """
    # Creates absolute directory path for downloaded data zip file.
    home_directory_path = os.getcwd()
    zip_file_path = "{}/data/raw_data/archive.zip".format(home_directory_path)

    # Creates the directory path.
    extracted_data_directory_path = check_directory_path_existence(
        "data/extracted_data/lgg_mri_segmentation"
    )

    # If file does not exist, then extracts files from the directory.
    if not os.path.exists(
        "{}/kaggle_3m/data.csv".format(extracted_data_directory_path)
    ):

        # Extracts files from downloaded data zip file into a directory.
        try:
            with zipfile.ZipFile(zip_file_path, "r") as zip_file:
                zip_file.extractall(extracted_data_directory_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "{} does not exist. Download data from "
                "'https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation', and place it in "
                "'data/raw_data' as 'archive.zip'.".format(zip_file_path)
            )
        print(
            "Finished extracting files from 'archive.zip' to {}.".format(
                extracted_data_directory_path
            )
        )
        print()


def load_dataset_file_paths() -> List[Dict[str, str]]:
    """Loads file paths of images & masks in the dataset.

    Loads file paths of images & masks in the dataset.

    Args:
        None.

    Returns:
        None.
    """
    home_directory_path = os.getcwd()
    extracted_data_directory_path = (
        "{}/data/extracted_data/lgg_mri_segmentation".format(home_directory_path)
    )

    # Lists patient's directory names from the extracted data.
    home_directory_path = os.getcwd()
    patients_directory_names = os.listdir(
        "{}/kaggle_3m".format(extracted_data_directory_path)
    )

    # Creates empty list to store image & mask file paths as records.
    file_paths = list()

    # Iterates across patients directory names.
    for directory_name in patients_directory_names:
        patient_directory_path = "{}/kaggle_3m/{}".format(
            extracted_data_directory_path, directory_name
        )

        # If directory name is not a directory or does not start with 'TCGA'.
        if not os.path.isdir(patient_directory_path) or not directory_name.startswith(
            "TCGA"
        ):
            continue

        # Lists file names in the patient directory path.
        image_file_names = os.listdir(patient_directory_path)

        # Iterates aross possible image ids.
        n_images = len(image_file_names) // 2
        image_id = 0
        while image_id <= n_images:
            image_file_path = "{}/{}_{}.tif".format(
                patient_directory_path, directory_name, image_id + 1
            )
            mask_file_path = "{}/{}_{}_mask.tif".format(
                patient_directory_path, directory_name, image_id + 1
            )

            # Checks if image & mask file paths are valid.
            if os.path.isfile(image_file_path) and os.path.isfile(mask_file_path):
                file_paths.append(
                    {
                        "image_file_paths": image_file_path,
                        "mask_file_paths": mask_file_path,
                    }
                )
            image_id += 1
    print(
        "No. of image & mask pair examples in the dataset: {}".format(len(file_paths))
    )
    print()
