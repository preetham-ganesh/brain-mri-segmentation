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
import numpy as np

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

    def predict(self, image_file_path: str) -> None:
        """Predicts if the brain MRI image has FLAIR abnormality.

        Predicts if the brain MRI image has FLAIR abnormality.

        Args:
            image_file_path: A string for the file path of the brain MRI image.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            image_file_path, str
        ), "Variable image_file_path of type 'str'."

        # Loads the image for the current image path.
        image = self.load_preprocess_image(image_file_path)

        # Predicts if the brain MRI image has FLAIR abnormality.
        prediction = self.model.predict(image)

        # Computes the predicted label based on the prediction.
        predicted_label = np.argmax(prediction[0].numpy()[0])

        # Prints the prediction results.
        if predicted_label == 0:
            print("Class: No FLAIR Abnormality detected.")
        else:
            print("Class: FLAIR Abnormality detected.")
        print(f"Confidence score: {float(prediction[0].numpy()[0][predicted_label])}")
        print()


def main():
    print()

    # Parses the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mv",
        "--model_version",
        type=str,
        required=True,
        help="Version of the model used to perform the prediction.",
    )
    parser.add_argument(
        "-ifp",
        "--image_file_path",
        type=str,
        required=True,
        help="Location where the image is located.",
    )
    args = parser.parse_args()

    # Creates an object for FlairAbnormalityClassification class.
    classification = FlairAbnormalityClassification(args.model_version)

    # Loads model configuration based on model version.
    classification.load_model_configuration()

    # Loads model based model configuration.
    classification.load_model()

    # Predicts mask for the current image using the current model.
    classification.predict(args.image_file_path)


if __name__ == "__main__":
    main()
