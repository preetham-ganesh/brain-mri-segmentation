import tensorflow as tf
from typing import Dict, Any, List


class ImageClassification(tf.keras.Model):
    """A tensorflow model which classifies if the brain MRI image has FLAIR abnormality."""

    def __init__(self, model_configuration: Dict[str, Any]) -> None:
        """Initializes the layers in the classification model, by adding various layers.

        Initializes the layers in the classification model, by adding various layers.

        Args:
            model_configuration: A dictionary for the configuration of the model.

        Returns:
            None.
        """
        super(ImageClassification, self).__init__()

        # Asserts type of input arguments.
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."

        # Initializes class variables.
        self.model_configuration = model_configuration
        self.model_layers = dict()

        # Iterates across layers arrangement in model configuration to add layers to the model.
        for name in self.model_configuration["model"]["layers"]["arrangement"]:
            config = self.model_configuration["model"]["layers"]["configuration"][name]

            # If layer's name is like 'mobilenet_', the MobileNet model is initialized based on layer configuration.
            if name.split("_")[0] == "mobilenet":
                self.model_layers[name] = tf.keras.applications.MobileNetV2(
                    include_top=config["include_top"],
                    weights=config["weights"],
                    input_shape=(
                        self.model_configuration["model"]["final_image_height"],
                        self.model_configuration["model"]["final_image_width"],
                        self.model_configuration["model"]["n_channels"],
                    ),
                )
                self.model_layers[name].trainable = config["trainable"]

            # If layer's name is like 'conv2d_', a Conv2D layer is initialized based on layer configuration.
            if name.split("_")[0] == "conv2d":
                self.model_layers[name] = tf.keras.layers.Conv2D(
                    filters=config["filters"],
                    kernel_size=config["kernel_size"],
                    padding=config["padding"],
                    strides=config["strides"],
                    activation=config["activation"],
                    name=name,
                )

            # If layer's name is like 'maxpool2d_', a MaxPool2D layer is initialized based on layer configuration.
            elif name.split("_")[0] == "maxpool2d":
                self.model_layers[name] = tf.keras.layers.MaxPool2D(
                    pool_size=config["pool_size"],
                    strides=config["strides"],
                    padding=config["padding"],
                    name=name,
                )

            # If layer's name is like 'dense_', a Dense layer is initialized based on layer configuration.
            elif name.split("_")[0] == "dense":
                self.model_layers[name] = tf.keras.layers.Dense(
                    units=config["units"], activation=config["activation"], name=name
                )

            # If layer's name is like 'dropout_', a Dropout layer is initialized based on layer configuration.
            elif name.split("_")[0] == "dropout":
                self.model_layers[name] = tf.keras.layers.Dropout(rate=config["rate"])

            # If layer's name is like 'flatten_', a Flatten layer is initialized.
            elif name.split("_")[0] == "flatten":
                self.model_layers[name] = tf.keras.layers.Flatten(name=name)

            # If layer's name is like 'flatten_', a Flatten layer is initialized.
            elif name.split("_")[0] == "globalaveragepool2d":
                self.model_layers[name] = tf.keras.layers.GlobalAveragePooling2D(
                    name=name
                )

    def call(
        self,
        inputs: List[tf.Tensor],
        training: bool = False,
        masks: List[tf.Tensor] = None,
    ) -> List[tf.Tensor]:
        """Input tensor is passed through the layers in the model.

        Input tensor is passed through the layers in the model.

        Args:
            inputs: A list for the inputs from the input batch.
            training: A boolean value for the flag of training/testing state.
            masks: A tensor for the masks from the input batch.

        Returns:
            A tensor for the processed output from the components in the layer.
        """
        # Asserts type & values of the input arguments.
        assert isinstance(inputs, list), "Variable inputs should be of type 'list'."
        assert isinstance(training, bool), "Variable training should be of type 'bool'."
        assert (
            isinstance(masks, list) or masks is None
        ), "Variable masks should be of type 'list' or masks should have value as 'None'."

        # Iterates across the layers arrangement, and predicts the output for each layer.
        x = inputs[0]
        for name in self.model_configuration["model"]["layers"]["arrangement"]:
            # If layer's name is like 'dropout_', the following output is predicted.
            if name.split("_")[0] == "dropout":
                x = self.model_layers[name](x, training=training)

            # Else, the following output is predicted.
            else:
                x = self.model_layers[name](x)
        return [x]

    def build_graph(self) -> tf.keras.Model:
        """Builds plottable graph for the model.

        Builds plottable graph for the model.

        Args:
            None.

        Returns:
            A tensorflow model based on image height, width & n_channels in the model configuration.
        """
        # Creates the input layer using the model configuration.
        inputs = [
            tf.keras.layers.Input(
                shape=(
                    self.model_configuration["model"]["final_image_height"],
                    self.model_configuration["model"]["final_image_width"],
                    self.model_configuration["model"]["n_channels"],
                )
            )
        ]
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs, False, None))
