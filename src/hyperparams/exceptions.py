
class SetsNotEqualException(ValueError):
    def __init__(self, train_size, val_size, test_size, set_size) -> None:
        super().__init__(f'Set size: {set_size} is not equal to sum of train, '
                         f'val, test sets: {train_size} + {val_size} + {test_size}.'
                         f'{set_size} != {train_size + val_size + test_size}')


class HyperparameterUnderZeroException(ValueError):
    def __init__(self) -> None:
        super().__init__("Hyperparameter value cannot be value under zero.")
