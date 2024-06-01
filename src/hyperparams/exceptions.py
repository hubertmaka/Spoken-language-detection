class SetsNotEqualException(ValueError):
    """
    Exception raised when the total size of the dataset does not equal the sum
    of train, validation, and test set sizes.

    Attributes:
        train_size (int): Size of the training set.
        val_size (int): Size of the validation set.
        test_size (int): Size of the test set.
        set_size (int): Total size of the dataset.

    Raises:
        ValueError: If the total size of the dataset is not equal to the sum of
                    the sizes of the train, validation, and test sets.
    """

    def __init__(self, train_size, val_size, test_size, set_size) -> None:
        super().__init__(f'Set size: {set_size} is not equal to sum of train, '
                         f'val, test sets: {train_size} + {val_size} + {test_size}.'
                         f'{set_size} != {train_size + val_size + test_size}')


class HyperparameterUnderZeroException(ValueError):
    """
    Exception raised when a hyperparameter value is set below zero.

    Raises:
        ValueError: If a hyperparameter value is less than zero.
    """

    def __init__(self) -> None:
        super().__init__("Hyperparameter value cannot be value under zero.")
