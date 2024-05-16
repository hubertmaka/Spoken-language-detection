import tensorflow as tf
import random
import librosa


class ProcessAudio:
    def __init__(self, audio_tensor: tf.Tensor, sample_rate: int):
        self._audio_tensor = audio_tensor
        self._sample_rate = sample_rate
        self._time_probes = self._audio_tensor.shape[0]
        self._audio_np = self._audio_tensor.numpy()

    def change_amplitude_rand(self, min_increase: float = 0.5, max_increase: float = 2.0):
        increased_amp = self._audio_tensor * random.uniform(min_increase, max_increase)
        return tf.convert_to_tensor(increased_amp, dtype=tf.float32)

    # TODO: do zmiany
    def normalize_audio(self) -> tf.Tensor:
        max_amplitude = tf.reduce_max(tf.abs(self._audio_tensor))
        normalized_wav = self._audio_tensor / max_amplitude  # Normalizacja do zakresu [-1, 1]
        return normalized_wav

    def add_noise_rand(self, min_level: float = 0.1, max_level: float = 1.5) -> tf.Tensor:
        noise_level = random.uniform(min_level, max_level)
        return self._audio_tensor + tf.random.normal(tf.shape(self._audio_tensor), mean=0.0, stddev=noise_level)

    def time_masking_rand(self, min_mask: int = 1000, max_mask: int = 5000) -> tf.Tensor:
        if tf.rank(self._audio_tensor) != 1:
            return self._audio_tensor

        max_mask_length = random.randint(min_mask, max_mask)

        if tf.shape(self._audio_tensor)[0] <= max_mask_length:
            return self._audio_tensor

        mask_length = tf.random.uniform([], maxval=max_mask_length, dtype=tf.int32)

        mask_start_max = tf.shape(self._audio_tensor)[0] - mask_length
        mask_start = tf.random.uniform([], maxval=mask_start_max, dtype=tf.int32)

        mask = tf.concat([
            tf.ones([mask_start]),
            tf.zeros([mask_length]),
            tf.ones([tf.shape(self._audio_tensor)[0] - mask_start - mask_length])
        ], axis=0)

        return self._audio_tensor * mask

    def change_pitch_rand(self, sample_rate=48_000, min_shift=2, max_shift=4) -> tf.Tensor:
        pitch_shift = random.randint(min_shift, max_shift)
        pitched_audio = librosa.effects.pitch_shift(self._audio_np, sr=sample_rate, n_steps=pitch_shift)
        return tf.convert_to_tensor(pitched_audio, dtype=tf.float32)

    def change_speed_rand(self, min_factor: float = 0.5, max_factor: float = 2.0) -> tf.Tensor:
        speed_factor = random.uniform(min_factor, max_factor)
        stretched_audio = librosa.effects.time_stretch(self._audio_np, rate=speed_factor)
        return tf.convert_to_tensor(stretched_audio, dtype=tf.float32)

    def create_spectrogram(self):
        spectrogram = tf.signal.stft(self._audio_tensor, frame_length=320, frame_step=32)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, axis=2)
        return spectrogram




