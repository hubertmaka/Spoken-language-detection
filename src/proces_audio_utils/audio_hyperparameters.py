import src.proces_audio_utils.exceptions as ex


class AudioHyperparameters:

    def __init__(self, max_client_id_amount: int, min_clip_duration_ms: int) -> None:
        if max_client_id_amount < 0 or min_clip_duration_ms < 0:
            raise ex.HyperparameterUnderZeroException()
        self._max_client_id_amount = max_client_id_amount
        self._min_clip_duration_ms = min_clip_duration_ms

    @property
    def max_client_id_amount(self) -> int:
        return self._max_client_id_amount

    @property
    def min_clip_duration_ms(self) -> int:
        return self._min_clip_duration_ms
