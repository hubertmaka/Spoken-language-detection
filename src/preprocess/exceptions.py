class NotEnoughProbesInSetError(Exception):
    """
    Exception raised when a dataset does not have the required number of probes.

    This error is used to indicate when the actual number of data points in a dataset is less than the required
    minimum, specifying the dataset and the shortfall in number.

    Attributes:
        set_name (str): The name of the dataset that is lacking probes.
        probes_diff (int): The number of missing probes needed to meet the requirement.

    Parameters:
        set_name (str): The name of the dataset.
        probes_diff (int): The shortfall in the number of probes.

    Raises:
        Exception: With a message indicating which set is missing how many probes.
    """

    def __init__(self, set_name: str, probes_diff: int) -> None:
        """
        Initializes the exception with the dataset name and the number of missing probes.

        Parameters:
            set_name (str): The name of the dataset that is short of probes.
            probes_diff (int): The number of probes that are missing from the dataset.
        """
        super().__init__(f"Not enough probes for {set_name}. Missing probes: {probes_diff}")
