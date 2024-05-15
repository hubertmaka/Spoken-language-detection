import tensorflow as tf
from paths_info import PathsInfo as hparams
import random

class ProcessAudio:
    def __init__(self, wav_tensor: tf.Tensor, sample_rate: int):
        self._wav_tensor = wav_tensor
        self._sample_rate = sample_rate
        self._time_probes = self._wav_tensor.shape[0]

    def _cut_audio(self) -> tf.Tensor:
        overlap = int((self._time_probes - (hparams.MIN_CLIP_DURATION_MS / 1000) * self._sample_rate) / 2)
        return self._wav_tensor[overlap:(self._time_probes - overlap)]

    def _add_zeros(self) -> tf.Tensor:
        missing_probes_side = int((hparams.MIN_CLIP_DURATION_MS / 1000 * self._sample_rate - self._time_probes) // 2)
        return tf.pad(self._wav_tensor.numpy(), [[missing_probes_side, missing_probes_side]])

    def align_probes(self) -> tf.Tensor:
        expected_probes = (hparams.MIN_CLIP_DURATION_MS / 1000) * self._sample_rate
        current_probes = self._time_probes
        if expected_probes > current_probes:
            return self._add_zeros()
        if expected_probes < current_probes:
            return self._cut_audio()
        if expected_probes == current_probes:
            return self._wav_tensor

    # def change_amplitude_randomize(self, min: float = 0.5, max: float = 5):
    #     return self._wav_tensor * random.uniform(min, max)



