import os
import time

import tensorflow as tf
import mlflow

from src.utils import load_json_file, check_directory_path_existence
from src.flair_abnormality_segmentation.dataset import Dataset
from src.flair_abnormality_segmentation.model import UNet


class Train(object):
    """Trains the segmentation model based on the configuration."""

    def __init__(self, model_version: str) -> None:
        """Creates object attributes for the Train class.

        Creates object attributes for the Train class.

        Args:
            model_version: A string for the version of the current model.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(model_version, str), "Variable model_version of type 'str'."

        # Initalizes class variables.
        self.model_version = model_version
        self.best_validation_loss = None

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

    def load_dataset(self) -> None:
        """Loads file paths of images & masks in the dataset.

        Loads file paths of images & masks in the dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Initializes object for the Dataset class.
        self.dataset = Dataset(self.model_configuration)

        # Loads the metadata for the images in the dataset.
        self.dataset.load_dataset_file_paths()

        # Splits file paths into new train, validation & test file paths.
        self.dataset.split_dataset()

        # Converts split data into tensor dataset & slices them based on batch size.
        self.dataset.shuffle_slice_dataset()

    def load_model(self) -> None:
        """Loads model & other utilies based on model configuration.

        Loads model & other utilies based on model configuration.

        Args:
            None.

        Returns:
            None.
        """
        # Loads model for current model configuration.
        self.model = UNet(self.model_configuration)

        # Loads the optimizer.
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.model_configuration["model"]["learning_rate"]
        )

        # Creates checkpoint manager for the neural network model.
        self.checkpoint_directory_path = os.path.join(
            self.home_directory_path,
            "models/flair_abnormality_segmentation",
            f"v{self.model_version}",
            "checkpoints",
        )
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, model=self.model
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.checkpoint_directory_path, max_to_keep=1
        )
        print("Finished loading model for current configuration.")
        print()

    def generate_model_summary_and_plot(self, plot: bool) -> None:
        """Generates summary & plot for loaded model.

        Generates summary & plot for loaded model.

        Args:
            pool: A boolean value to whether generate model plot or not.

        Returns:
            None.
        """
        # Builds plottable graph for the model.
        model = self.model.build_graph()

        # Compiles the model to log the model summary.
        model_summary = list()
        model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary = "\n".join(model_summary)
        print(model_summary)
        mlflow.log_text(model_summary, f"v{self.model_version}/model_summary.txt")

        # Creates the following directory path if it does not exist.
        self.reports_directory_path = check_directory_path_existence(
            f"models/flair_abnormality_segmentation/v{self.model_version}/reports"
        )

        # Plots the model & saves it as a PNG file.
        if plot:
            tf.keras.utils.plot_model(
                model,
                os.path.join(self.reports_directory_path, "model_plot.png"),
                show_shapes=True,
                show_layer_names=True,
                expand_nested=True,
            )

            # Logs the saved model plot PNG file.
            mlflow.log_artifact(
                os.path.join(self.reports_directory_path, "model_plot.png"),
                f"v{self.model_version}",
            )

    def initialize_metric_trackers(self) -> None:
        """Initializes trackers which computes the mean of all metrics.

        Initializes trackers which computes the mean of all metrics.

        Args:
            None.

        Returns:
            None.
        """
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.validation_loss = tf.keras.metrics.Mean(name="validation_loss")
        self.train_dice = tf.keras.metrics.Mean(name="train_dice_coefficient")
        self.validation_dice = tf.keras.metrics.Mean(name="validation_dice_coefficient")
        self.train_iou = tf.keras.metrics.Mean(name="train_iou")
        self.validation_iou = tf.keras.metrics.Mean(name="validation_iou")

    def compute_loss(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes loss for the current batch using actual & predicted values.

        Computes loss for the current batch using actual & predicted values.

        Args:
            target_batch: A tensor for target batch of generated mask images.
            predicted_batch: A tensor for batch of outputs predicted by the model for input batch.

        Returns:
            A tensor for the loss computed on comparing target & predicted batch.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."
        assert isinstance(
            predicted_batch, tf.Tensor
        ), "Variable predicted_batch should be of type 'tf.Tensor'."

        # Computes loss for current target & predicted batches.
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        current_loss = loss_object(target_batch, predicted_batch)
        return current_loss

    def compute_dice_coefficient(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes dice coefficient for the current batch using actual & predicted values.

        Computes dice coefficient for the current batch using actual & predicted values.

        Args:
            target_batch: A tensor for target batch of generated mask images.
            predicted_batch: A tensor for batch of outputs predicted by the model for input batch.

        Returns:
            A tensor for the dice coefficient computed on comparing target & predicted batch.
        """
        # Flattens the target and predicted batches.
        target_batch = tf.keras.layers.Flatten()(target_batch)
        predicted_batch = tf.keras.layers.Flatten()(predicted_batch)

        # Computes the intersection between the flattened target and predicted batches.
        intersection = tf.reduce_sum(target_batch * predicted_batch)
        smooth = 1e-15
        return (2.0 * intersection + smooth) / (
            tf.reduce_sum(target_batch) + tf.reduce_sum(predicted_batch) + smooth
        )

    def compute_intersection_over_union(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes IoU for the current batch using actual & predicted values.

        Computes IoU for the current batch using actual & predicted values.

        Args:
            target_batch: A tensor for target batch of generated mask images.
            predicted_batch: A tensor for batch of outputs predicted by the model for input batch.

        Returns:
            A tensor for the IoU computed on comparing target & predicted batch.
        """
        # Flattens the target and predicted batches.
        target_batch = tf.keras.layers.Flatten()(target_batch)
        predicted_batch = tf.keras.layers.Flatten()(predicted_batch)

        # Computes intersection & union for the target and predicted batch.
        intersection = tf.reduce_sum(target_batch * predicted_batch)
        union = (
            tf.reduce_sum(target_batch) + tf.reduce_sum(predicted_batch) - intersection
        )
        # Computes Intersection over Union metric.
        smooth = 1e-15
        iou = (intersection + smooth) / (union + smooth)
        return iou

    @tf.function
    def train_step(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> None:
        """Trains the model using input & target batches.

        Trains the model using input & target batches.

        Args:
            input_batch: A tensor for input batch of processed images.
            target_batch: A tensor for target batch of generated mask images.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            input_batch, tf.Tensor
        ), "Variable input_batch should be of type 'tf.Tensor'."
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."

        # Computes masked images for all input images in the batch, and computes batch loss.
        with tf.GradientTape() as tape:
            predicted_batch = self.model([input_batch], training=True, masks=None)[0]
            batch_loss = self.compute_loss(target_batch, predicted_batch)

        # Computes gradients using loss. Apply the computed gradients on model variables using optimizer.
        gradients = tape.gradient(batch_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Computes dice coefficient and iou score for the current batch.
        batch_dice = self.compute_dice_coefficient(target_batch, predicted_batch)
        batch_iou = self.compute_intersection_over_union(target_batch, predicted_batch)

        # Computes mean for loss, dice coefficient and iou score.
        self.train_loss(batch_loss)
        self.train_dice(batch_dice)
        self.train_iou(batch_iou)

    def validation_step(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> None:
        """Validates the model using input and target batches.

        Validates the model using input and target batches.

        Args:
            input_batch: A tensor for input batch of processed images.
            target_batch: A tensor for target batch of generated mask images.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            input_batch, tf.Tensor
        ), "Variable input_batch should be of type 'tf.Tensor'."
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."

        # Computes masked images for all input images in the batch.
        predicted_batch = self.model([input_batch], training=False, masks=None)[0]

        # Computes loss, dice coefficient & IoU for the target batch and predicted batch.
        batch_loss = self.compute_loss(target_batch, predicted_batch)
        batch_dice = self.compute_dice_coefficient(target_batch, predicted_batch)
        batch_iou = self.compute_intersection_over_union(target_batch, predicted_batch)

        # Computes mean for loss & accuracy.
        self.validation_loss(batch_loss)
        self.validation_dice(batch_dice)
        self.validation_iou(batch_iou)

    def reset_metrics_trackers(self) -> None:
        """Resets states for trackers before the start of each epoch.

        Resets states for trackers before the start of each epoch.

        Args:
            None.

        Returns:
            None.
        """
        self.train_loss.reset_state()
        self.validation_loss.reset_state()
        self.train_dice.reset_state()
        self.validation_dice.reset_state()
        self.train_iou.reset_state()
        self.validation_iou.reset_state()

    def train_model_per_epoch(self, epoch: int) -> None:
        """Trains the model using train dataset for current epoch.

        Trains the model using train dataset for current epoch.

        Args:
            epoch: An integer for the number of current epoch.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(epoch, int), "Variable epoch should be of type 'int'."

        # Iterates across batches in the train dataset.
        for batch, (image_file_paths, mask_file_paths) in enumerate(
            self.dataset.train_dataset.take(self.dataset.n_train_steps_per_epoch)
        ):
            batch_start_time = time.time()

            # Loads input & target batch images for file paths in current batch.
            input_batch, target_batch = self.dataset.load_input_target_batches(
                list(image_file_paths.numpy()), list(mask_file_paths.numpy())
            )

            # Trains the model using the current input and target batch.
            self.train_step(input_batch, target_batch)
            batch_end_time = time.time()
            print(
                f"Epoch={epoch + 1}, Batch={batch}, Train loss={self.train_loss.result().numpy():.3f}, "
                + f"Train coefficient={self.train_dice.result().numpy():.3f}, "
                + f"Train IoU={self.train_iou.result().numpy():.3f}, "
                + f"Time taken={(batch_end_time - batch_start_time):.3f} sec."
            )

        # Logs train metrics for current epoch.
        mlflow.log_metrics(
            {
                "train_loss": self.train_loss.result().numpy(),
                "train_dice_coefficient": self.train_dice.result().numpy(),
                "train_iou": self.train_iou.result().numpy(),
            },
            step=epoch,
        )
        print("")

    def validate_model_per_epoch(self, epoch: int) -> None:
        """Validates the model using the current validation dataset.

        Validates the model using the current validation dataset.

        Args:
            epoch: An integer for the number of current epoch.

        Returns:
            None
        """
        # Asserts type & value of the arguments.
        assert isinstance(epoch, int), "Variable epoch should be of type 'int'."

        # Iterates across batches in the validation dataset.
        for batch, (image_file_paths, mask_file_paths) in enumerate(
            self.dataset.validation_dataset.take(
                self.dataset.n_validation_steps_per_epoch
            )
        ):
            batch_start_time = time.time()

            # Loads input & target batch images for file paths in current batch.
            input_batch, target_batch = self.dataset.load_input_target_images(
                list(image_file_paths.numpy()), list(mask_file_paths.numpy())
            )

            # Validates the model using the current input and target batch.
            self.validation_step(input_batch, target_batch)
            batch_end_time = time.time()
            print(
                f"Epoch={epoch + 1}, Batch={batch}, Validation loss={self.validation_loss.result().numpy():.3f}, "
                + f"Validation coefficient={self.validation_dice.result().numpy():.3f}, "
                + f"Validation IoU={self.validation_iou.result().numpy():.3f}, "
                + f"Time taken={(batch_end_time - batch_start_time):.3f} sec."
            )

        # Logs train metrics for current epoch.
        mlflow.log_metrics(
            {
                "validation_loss": self.train_loss.result().numpy(),
                "validation_dice_coefficient": self.train_dice.result().numpy(),
                "validation_iou": self.train_iou.result().numpy(),
            },
            step=epoch,
        )
        print("")

    def save_model(self) -> None:
        """Saves the model after checking performance metrics in current epoch.

        Saves the model after checking performance metrics in current epoch.

        Args:
            None.

        Returns:
            None.
        """
        self.manager.save()
        print(f"Checkpoint saved at {self.checkpoint_directory_path}.")

    def early_stopping(self) -> bool:
        """Stops the model from learning further if the performance has not improved from previous epoch.

        Stops the model from learning further if the performance has not improved from previous epoch.

        Args:
            None.

        Returns:
            None.
        """
        # If epoch = 1, then best validation loss is replaced with current validation loss, & the checkpoint is saved.
        if self.best_validation_loss is None:
            self.patience_count = 0
            self.best_validation_loss = round(
                float(self.validation_loss.result().numpy()), 3
            )
            self.save_model()

        # If best validation loss is higher than current validation loss, the best validation loss is replaced with
        # current validation loss, & the checkpoint is saved.
        elif self.best_validation_loss > round(
            float(self.validation_loss.result().numpy()), 3
        ):
            self.patience_count = 0
            print(
                f"Best validation loss changed from {self.best_validation_loss} to "
                + f"{self.validation_loss.result().numpy():.3f}"
            )
            self.best_validation_loss = round(
                float(self.validation_loss.result().numpy()), 3
            )
            self.save_model()

        # If best validation loss is not higher than the current validation loss, then the number of times the model
        # has not improved is incremented by 1.
        elif self.patience_count < self.model_configuration["model"]["patience_count"]:
            self.patience_count += 1
            print("Best validation loss did not improve.")
            print("Checkpoint not saved.")

        # If the number of times the model did not improve is greater than 4, then model is stopped from training.
        else:
            return False
        return True

    def fit(self) -> None:
        """Trains & validates the loaded model using train & validation dataset.

        Trains & validates the loaded model using train & validation dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Initializes trackers which computes the mean of all metrics.
        self.initialize_metric_trackers()

        # Iterates across epochs for training the neural network model.
        for epoch in range(self.model_configuration["model"]["epochs"]):
            epoch_start_time = time.time()

            # Resets states for trackers before the start of each epoch.
            self.reset_metrics_trackers()

            # Trains the model using batces in the train dataset.
            self.train_model_per_epoch(epoch)

            # Validates the model using batches in the validation dataset.
            self.validate_model_per_epoch(epoch)

            epoch_end_time = time.time()
            print(
                f"Epoch={epoch + 1}, Train loss={self.train_loss.result().numpy():.3f}, "
                + f"Validation loss={self.validation_loss.result().numpy():.3f}, "
                + f"Train dice coefficient={self.train_dice.result().numpy():.3f}, "
                + f"Validation dice coefficient={self.validation_dice.result().numpy():.3f}, "
                + f"Train IoU={self.train_iou.result().numpy():.3f}, "
                + f"Validation IoU={self.validation_iou.result().numpy():.3f}, "
                + f"Time taken={(epoch_end_time - epoch_start_time):.3f} sec."
            )

            # Stops the model from learning further if the performance has not improved from previous epoch.
            model_training_status = self.early_stopping()
            if not model_training_status:
                print(
                    "Model did not improve after 4th time. Model stopped from training further."
                )
                print()
                break
            print()

    def test_model(self) -> None:
        """Tests the trained model using the test dataset.

        Tests the trained model using the test dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Resets states for validation metrics.
        self.reset_metrics_trackers()

        # Restore latest saved checkpoint if available.
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_directory_path)
        ).assert_consumed()

        # Iterates across batches in the train dataset.
        for batch, (images, labels) in enumerate(
            self.dataset.test_dataset.take(self.dataset.n_test_steps_per_epoch)
        ):
            # Loads input & target sequences for current batch as tensors.
            input_batch, target_batch = self.dataset.load_input_target_batches(
                list(images.numpy()), list(labels.numpy())
            )

            # Tests the model using the current input and target batch.
            self.validation_step(input_batch, target_batch)

        print(f"Test loss: {self.validation_loss.result().numpy():.3f}.")
        print(f"Test dice coefficient: {self.validation_dice.result().numpy():.3f}")
        print(f"Test IoU: {self.validation_iou.result().numpy():.3f}")
        print()

        # Logs test metrics for current epoch.
        mlflow.log_metrics(
            {
                "test_loss": self.validation_loss.result().numpy(),
                "test_dice_coefficient": self.validation_dice.result().numpy(),
                "test_iou": self.validation_iou.result().numpy(),
            }
        )
