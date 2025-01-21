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
import numpy as np
import skimage


from src.utils import load_json_file, check_directory_path_existence
from src.flair_abnormality_segmentation.dataset import Dataset
from src.preprocess_data import load_image

from typing import List, Dict, Any


class FlairAbnormalitySegmentation(object):
    """Predicts segmentation mask for FLAIR abnormality in brain MRI images."""

    def __init__(self, model_version: str) -> None:
        """Creates object attributes for the FlairAbnormalitySegmentation class.

        Creates object attributes for the FlairAbnormalitySegmentation class.

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
            self.home_directory_path, "configs/flair_abnormality_segmentation"
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
        exported_model = tf.saved_model.load(
            os.path.join(
                self.home_directory_path,
                f"models/flair_abnormality_segmentation/v{self.model_version}/serialized",
            )
        )

        # Get the callable signature (default is "serving_default")
        self.model = exported_model.signatures["serving_default"]

        # Initializes object for the Dataset class.
        self.dataset = Dataset(self.model_configuration)

    def load_preprocess_input_image(self, image_file_path: str) -> List[np.ndarray]:
        """Loads & preprocesses image based on segmentation model requirements.

        Loads & preprocesses image based on segmentation model requirements.

        Args:
            image_file_path: A string for the location of the image.

        Returns:
            A list of NumPy arrays for the original & processed image as input to the model.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            image_file_path, str
        ), "Variable image_file_path should be of type 'str'."

        # Loads the image for the current image path.
        image = load_image(image_file_path)

        # Thresholds image to have better distinction of regions in image.
        model_input_image = self.dataset.threshold_image(image)

        # Casts input image to float32 and normalizes the image from [0, 255] range to [0, 1] range.
        model_input_image = tf.convert_to_tensor(model_input_image, dtype=tf.float32)
        model_input_image = model_input_image / 255.0

        # Adds an extra dimension to the image.
        model_input_image = tf.expand_dims(model_input_image, axis=0)
        return [image, model_input_image]

    def postprocess_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """Converts the prediction from the segmentation model output into an image.

        Converts the prediction from the segmentation model output into an image.

        Args:
            prediction: A NumPy array for the prediction output from the model for the input image.

        Returns:
            A NumPy array for the processed version of prediction output.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            prediction, np.ndarray
        ), "Variable prediction should be of type 'np.ndarray'."

        # Removes 0th and 2nd dimension from the predicted image.
        predicted_image = np.squeeze(prediction, axis=0)
        predicted_image = np.squeeze(predicted_image, axis=-1)

        # De-normalizes predicted image from [0, 1] to [0, 255].
        predicted_image *= 255.0

        # Thresholds the predicted image to convert into black & white image, and type casts it to uint8
        predicted_image = self.dataset.threshold_image(predicted_image)
        predicted_image = predicted_image.astype(np.uint8)
        return predicted_image

    def predict(self, image_file_path: str) -> None:
        """Predicts mask for the current image using the current model.

        Predicts mask for the current image using the current model.

        Args:
            image_file_path: A string for the location where the image is located.

        Returns:
            None
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            image_file_path, str
        ), "Variable image_file_path should be of type 'str'."

        # Loads & preprocesses image based on segmentation model requirements.
        image, model_input_image = self.load_preprocess_input_image(image_file_path)

        # Predicts the class for each pixel in the current image input.
        prediction = self.model(model_input_image)

        # Converts the prediction from the segmentation model into an image.
        predicted_image = self.postprocess_prediction(prediction["output_0"].numpy())

        # Creates the following path if it does not exist.
        self.predicted_images_directory_path = check_directory_path_existence(
            f"models/flair_abnormality_segmentation/v{self.model_version}/predictions"
        )

        # Computes number of images predicted using this model.
        images_predicted = [
            name
            for name in os.listdir(self.predicted_images_directory_path)
            if name[0] != "."
        ]
        n_images_predicted = int(len(images_predicted) / 2)

        # Saves input, and predicted images.
        skimage.io.imsave(
            f"{self.predicted_images_directory_path}/{n_images_predicted}_input.png",
            image,
        )
        print(
            f"Original image saved at {self.predicted_images_directory_path}/{n_images_predicted}_input.png."
        )
        skimage.io.imsave(
            f"{self.predicted_images_directory_path}/{n_images_predicted}_predicted.png",
            predicted_image,
        )
        print(
            f"Predicted mask image saved at {self.predicted_images_directory_path}/{n_images_predicted}_predicted.png."
        )
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

    # Creates an object for Predict class.
    segmentation = FlairAbnormalitySegmentation(args.model_version)

    # Loads model configuration based on model name & model version.
    segmentation.load_model_configuration()

    # Loads model based model configuration.
    segmentation.load_model()

    # Predicts mask for the current image using the current model.
    segmentation.predict(args.image_file_path)


if __name__ == "__main__":
    main()
