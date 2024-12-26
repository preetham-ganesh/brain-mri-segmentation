import os
import sys
import warnings
import logging
import argparse


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.FATAL)


import tensorflow as tf
import cv2

from src.utils import load_json_file
from src.flair_abnormality_classification.dataset import Dataset
from src.preprocess_data import load_image


class FlairAbnormalityClassification(object):
    """Predicts whether is FLAIR abnormality in brain MRI images."""

    def __init__(self, model_version: str) -> None:
        """Creates object attributes for the FlairAbnormalityClassification class.

        Creates object attributes for the FlairAbnormalityClassification class.

        Args:
            model_version: A string for the version of the model should be used for prediction.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(model_version, str), "Variable model_version of type 'str'."

        # Initalizes class variables.
        self.model_version = model_version

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

    def load_model(self) -> None:
        """Loads model & other utilities for prediction.

        Loads model & other utilities for prediction.

        Args:
            None.

        Returns:
            None.
        """
        # Loads the tensorflow serialized model using model name & version.
        self.home_directory_path = os.getcwd()
        self.model = tf.saved_model.load(
            os.path.join(
                self.home_directory_path,
                f"models/flair_abnormality_classification/v{self.model_version}/serialized",
            )
        )

        # Initializes object for the Dataset class.
        self.dataset = Dataset(self.model_configuration)

    def load_preprocess_image(self, image_file_path: str) -> tf.Tensor:
        """Preprocesses the image for prediction.

        Preprocesses the image for prediction.

        Args:
            image_file_path: A string for the file path of the brain MRI image.

        Returns:
            A tensor for the preprocessed image.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            image_file_path, str
        ), "Variable image_file_path of type 'str'."

        # Loads the image for the current image path.
        image = load_image(image_file_path)

        # Preprocesses the image for prediction.
        if (
            image.shape[0] > self.model_configuration["model"]["final_image_height"]
            or image.shape[1] > self.model_configuration["model"]["final_image_width"]
        ):
            image = cv2.resize(
                image,
                (
                    self.model_configuration["model"]["final_image_height"],
                    self.model_configuration["model"]["final_image_width"],
                ),
            )

        # Converts the image to tensor.
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        # Expands the dimensions of the image in the first axis.
        image = tf.expand_dims(image, axis=0)
        return image
