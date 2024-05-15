import src.proces_audio_utils.exceptions as ex


class SetHyperparameters:

    def __init__(self, set_size: int) -> None:
        self._set_size = set_size
        self._train_size = 0
        self._val_size = 0
        self._test_size = 0

        self._set_train_val_test_sizes()
        self._check_property()

    def _set_train_val_test_sizes(self) -> None:
        self._train_size = int(self._set_size * 0.6)
        self._val_size = int((self._set_size - self._train_size) // 2)
        self._test_size = int(self._set_size - self._train_size - self._val_size)

    def _check_property(self) -> None:
        if (self._train_size + self._val_size + self._test_size) != self._set_size:
            raise ex.SetsNotEqualException(self._train_size, self._val_size, self._test_size, self._set_size)

    @property
    def set_size(self) -> int:
        return self._set_size

    @property
    def train_size(self) -> int:
        return self._train_size

    @property
    def val_size(self) -> int:
        return self._val_size

    @property
    def test_size(self) -> int:
        return self._test_size

    def print_train_val_test_sizes(self) -> None:
        print(f'Train size: {self._train_size}')
        print(f'Validation size: {self._val_size}')
        print(f'Test size: {self._test_size}')
        print(f'Sum of sizes: {self._train_size + self._val_size + self.test_size}')
