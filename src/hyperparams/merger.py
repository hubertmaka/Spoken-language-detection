from src.hyperparams.sample_hparams import AudioHyperparameters
from src.hyperparams.set_hparams import SetHyperparameters
from src.hyperparams.paths_info import PathsInfo


class HparamsMerger:
    def __init__(self, max_client_id_amount: int, min_clip_duration_ms: int, set_size: int, lang: str):
        self._audio_hyperparameters = AudioHyperparameters(max_client_id_amount, min_clip_duration_ms)
        self._set_hyperparameters = SetHyperparameters(set_size)
        self._paths_info = PathsInfo()
        self.lang = lang

    @property
    def audio_hyperparameters(self) -> AudioHyperparameters:
        return self._audio_hyperparameters

    @property
    def set_hyperparameters(self) -> SetHyperparameters:
        return self._set_hyperparameters

    @property
    def paths_info(self) -> PathsInfo:
        return self._paths_info