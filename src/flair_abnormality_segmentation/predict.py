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
