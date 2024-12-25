import os
import sys
import warnings
import argparse
import zipfile


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")


import numpy as np
import skimage
import cv2

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
    zip_file_path = os.path.join(home_directory_path, "data/raw_data/archive.zip")

    # Creates the directory path.
    extracted_data_directory_path = check_directory_path_existence(
        "data/extracted_data/lgg_mri_segmentation"
    )

    # If file does not exist, then extracts files from the directory.
    if not os.path.exists(
        os.path.join(extracted_data_directory_path, "/kaggle_3m/data.csv")
    ):

        # Extracts files from downloaded data zip file into a directory.
        try:
            with zipfile.ZipFile(zip_file_path, "r") as zip_file:
                zip_file.extractall(extracted_data_directory_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"{zip_file_path} does not exist. Download data from "
                "'https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation', and place it in "
                "'data/raw_data' as 'archive.zip'."
            )
        print(
            f"Finished extracting files from 'archive.zip' to {extracted_data_directory_path}."
        )
        print()


def load_dataset_file_paths() -> List[Dict[str, str]]:
    """Loads file paths of images & masks in the dataset.

    Loads file paths of images & masks in the dataset.

    Args:
        None.

    Returns:
        A list of dictionaries for image & mask file paths.
    """
    home_directory_path = os.getcwd()
    extracted_data_directory_path = os.path.join(
        home_directory_path, "data/extracted_data/lgg_mri_segmentation"
    )

    # Lists patient's directory names from the extracted data.
    home_directory_path = os.getcwd()
    patients_directory_names = os.listdir(
        os.path.join(extracted_data_directory_path, "kaggle_3m")
    )

    # Creates empty list to store image & mask file paths as records.
    file_paths = list()

    # Iterates across patients directory names.
    for directory_name in patients_directory_names:
        patient_directory_path = os.path.join(
            extracted_data_directory_path, "kaggle_3m", directory_name
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
            image_file_path = os.path.join(
                patient_directory_path, f"{directory_name}_{image_id + 1}.tif"
            )
            mask_file_path = os.path.join(
                patient_directory_path, f"{directory_name}_{image_id + 1}_mask.tif"
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
    print(f"No. of image & mask pair examples in the dataset: {len(file_paths)}")
    print()
    return file_paths


def load_image(image_file_path: str) -> np.ndarray:
    """Loads the image for the current image path.

    Loads the image for the current image path.

    Args:
        image_file_path: A string for the location where the image is located.

    Returns:
        A NumPy array for the image loaded from the file path.
    """
    # Checks type & values of arguments.
    assert isinstance(
        image_file_path, str
    ), "Variable image_file_path should be of type 'str'."

    # Loads the image for the current image path.
    image = skimage.io.imread(image_file_path)
    return image


def preprocess_dataset(file_paths: List[Dict[str, str]], dataset_version: str) -> None:
    """Preprocesses the images in the dataset to split them into 2 categories, abnormality & no abnormality.

    Preprocesses the images in the dataset to split them into 2 categories, abnormality & no abnormality.

    Args:
        file_paths: A list of dictionaries for image & mask file paths in the dataset.
        dataset_version: A string for the version by which the dataset should be saved as.

    Returns:
        None.
    """
    # Checks if the following directory path exists.
    no_abnormality_images_directory_path = check_directory_path_existence(
        f"data/processed_data/lgg_mri_segmentation/v{dataset_version}/no_abnormality/images"
    )
    no_abnormality_masks_directory_path = check_directory_path_existence(
        f"data/processed_data/lgg_mri_segmentation/v{dataset_version}/no_abnormality/masks"
    )
    abnormality_images_directory_path = check_directory_path_existence(
        f"data/processed_data/lgg_mri_segmentation/v{dataset_version}/abnormality/images"
    )
    abnormality_masks_directory_path = check_directory_path_existence(
        f"data/processed_data/lgg_mri_segmentation/v{dataset_version}/abnormality/masks"
    )

    # Iterates across files in the dataset.
    n_files = len(file_paths)
    abnormality_count, no_abnormality_count = 0, 0
    for f_id in range(n_files):

        # Loads the image for the current image path.
        image = load_image(file_paths[f_id]["image_file_paths"])
        mask = load_image(file_paths[f_id]["mask_file_paths"])

        # Checks if mask image has pixel value apart from 0. Saves image & mask accordingly.
        if np.any(mask != 0):
            cv2.imwrite(
                os.path.join(abnormality_images_directory_path, f"{f_id}.png"), image
            )
            cv2.imwrite(
                os.path.join(abnormality_masks_directory_path, f"{f_id}.png"), mask
            )
            abnormality_count += 1
        else:
            cv2.imwrite(
                os.path.join(no_abnormality_images_directory_path, f"{f_id}.png"), image
            )
            cv2.imwrite(
                os.path.join(no_abnormality_masks_directory_path, f"{f_id}.png"), mask
            )
            no_abnormality_count += 1

        if f_id % 1000 == 0 and f_id != 0:
            print(
                f"Finished processing {round((f_id / n_files) * 100, 3)}% images & masks in the dataset."
            )
    print()
    print(f"No. of images in the no abnormality class: {no_abnormality_count}")
    print(f"No. of images in the abnormality class: {abnormality_count}")
    print()


def main():
    print()
    # Parses the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dv",
        "--dataset_version",
        type=str,
        required=True,
        help="Version by which the processed dataset should be saved as.",
    )
    args = parser.parse_args()

    # Extracts files from downloaded data zip file.
    extract_data_from_zip_file()

    # Loads file paths of images & masks in the dataset.
    file_paths = load_dataset_file_paths()

    # Preprocesses the images in the dataset to split them into 2 categories, abnormality & no abnormality.
    preprocess_dataset(file_paths, args.dataset_version)


if __name__ == "__main__":
    main()
