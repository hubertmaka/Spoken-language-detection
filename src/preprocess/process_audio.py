import tensorflow as tf
from src.preprocess.preprocess import Preprocess


class ProcessAudio:
    def __init__(self, wav_tensor: tf.Tensor, sample_rate: int):
        self._wav_tensor = wav_tensor
        self._sample_rate = sample_rate
        self._time_probes = self._wav_tensor.shape[0]

    def _cut_audio(self, min_clip_duration_ms) -> tf.Tensor:
        overlap = int((self._time_probes - (min_clip_duration_ms / 1000) * self._sample_rate) / 2)
        return self._wav_tensor[overlap:(self._time_probes - overlap)]

    def _add_zeros(self, min_clip_duration_ms) -> tf.Tensor:
        missing_probes_side = int((min_clip_duration_ms / 1000 * self._sample_rate - self._time_probes) // 2)
        return tf.pad(self._wav_tensor.numpy(), [[missing_probes_side, missing_probes_side]])

    def align_probes(self, min_clip_duration_ms) -> tf.Tensor:
        expected_probes = (min_clip_duration_ms / 1000) * self._sample_rate
        current_probes = self._time_probes
        if expected_probes > current_probes:
            return self._add_zeros(min_clip_duration_ms)
        if expected_probes < current_probes:
            return self._cut_audio(min_clip_duration_ms)
        if expected_probes == current_probes:
            return self._wav_tensor

    def load_and_align_probes(self, file_path, min_clip_duration_ms):
        wav = Preprocess.load_audio(file_path)
        expected_probes = int((min_clip_duration_ms / 1000) * self._sample_rate)
        print(expected_probes)
        current_probes = wav.shape[0]
        print(current_probes)
        if expected_probes > current_probes:
            print("Add zeros")
            return self._add_zeros(min_clip_duration_ms)
        elif expected_probes < current_probes:
            print("Cut wav")
            return self._cut_audio(min_clip_duration_ms)
        return tf.convert_to_tensor(wav, dtype=tf.float32)


    # def change_amplitude_randomize(self, min: float = 0.5, max: float = 5):
    #     return self._wav_tensor * random.uniform(min, max)



