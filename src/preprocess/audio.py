import tensorflow as tf
import random
import librosa
import tensorflow_io as tfio


class ProcessAudio:
    def __init__(self, audio_tensor: tf.Tensor, sample_rate: int):
        self._audio_tensor = audio_tensor
        self._sample_rate = sample_rate
        self._time_probes = self._audio_tensor.shape[0]

    def change_amplitude_rand(self, *, min_increase: float = 0.5, max_increase: float = 2.0):
        increased_amp = self._audio_tensor * random.uniform(min_increase, max_increase)
        return tf.convert_to_tensor(increased_amp, dtype=tf.float32)

    def _remove_dc_offset(self) -> tf.Tensor:
        mean_val = tf.reduce_mean(self._audio_tensor)
        return self._audio_tensor - mean_val

    def _reduce_by_3db(self) -> tf.Tensor:
        return self._audio_tensor * 0.707

    def normalize_audio(self):
        self._audio_tensor = self._remove_dc_offset()
        self._audio_tensor = self._reduce_by_3db()
        return self._audio_tensor

    def add_noise_rand(self, *, min_level: float = 0.1, max_level: float = 1.5) -> tf.Tensor:
        noise_level = random.uniform(min_level, max_level)
        return self._audio_tensor + tf.random.normal(tf.shape(self._audio_tensor), mean=0.0, stddev=noise_level)

    def time_masking_rand(self, *, min_mask: int = 1000, max_mask: int = 5000) -> tuple[tf.Tensor, int]:
        if tf.rank(self._audio_tensor) != 1:
            return self._audio_tensor, self._sample_rate

        max_mask_length = random.randint(min_mask, max_mask)

        if tf.shape(self._audio_tensor)[0] <= max_mask_length:
            return self._audio_tensor, self._sample_rate

        mask_length = tf.random.uniform([], maxval=max_mask_length, dtype=tf.int32)

        mask_start_max = tf.shape(self._audio_tensor)[0] - mask_length
        mask_start = tf.random.uniform([], maxval=mask_start_max, dtype=tf.int32)

        mask = tf.concat([
            tf.ones([mask_start]),
            tf.zeros([mask_length]),
            tf.ones([tf.shape(self._audio_tensor)[0] - mask_start - mask_length])
        ], axis=0)

        return self._audio_tensor * mask, self._sample_rate

    def change_pitch_rand_librosa(self, *, sample_rate: int, min_shift: int = 2, max_shift: int = 4) -> tf.Tensor:
        pitch_shift = random.randint(min_shift, max_shift)
        audio = self._audio_tensor.numpy()
        pitched_audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift)
        return tf.convert_to_tensor(pitched_audio, dtype=tf.float32)

    def change_pitch_rand(self, *, sample_rate: int, min_shift: int = -2, max_shift: int = 4) -> tuple[tf.Tensor, int]:
        pitch_shift = random.randint(min_shift, max_shift)
        new_sample_rate = int(sample_rate * (2 ** (pitch_shift / 12.0)))
        resampled_audio = tfio.audio.resample(self._audio_tensor, rate_in=sample_rate, rate_out=new_sample_rate)
        return resampled_audio, new_sample_rate

    def change_speed_rand_librosa(self, *, min_factor: float = 0.5, max_factor: float = 2.0) -> tf.Tensor:
        speed_factor = random.uniform(min_factor, max_factor)
        audio = self._audio_tensor.numpy()
        stretched_audio = librosa.effects.time_stretch(audio, rate=speed_factor)
        return tf.convert_to_tensor(stretched_audio, dtype=tf.float32)

    def time_shift_rand(self, min_shift: int = -1000, max_shift: int = 1000) -> tuple[tf.Tensor, int]:
        shift = random.randint(min_shift, max_shift)
        if shift > 0:
            shifted_audio = tf.concat([tf.zeros([shift], dtype=self._audio_tensor.dtype), self._audio_tensor[:-shift]], axis=0)
        elif shift < 0:
            shifted_audio = tf.concat([self._audio_tensor[-shift:], tf.zeros([-shift], dtype=self._audio_tensor.dtype)], axis=0)
        else:
            shifted_audio = self._audio_tensor
        return shifted_audio, self._sample_rate

    def fade_rand(self, fade_in_min: int = 900, fade_in_max: int = 1100, fade_out_min: int = 1900, fade_out_max: int = 2100) -> tf.Tensor:
        fade_in = random.randint(fade_in_min, fade_in_max)
        fade_out = random.randint(fade_out_min, fade_out_max)
        return tfio.audio.fade(self._audio_tensor, fade_in, fade_out, mode='logarithmic')

    def create_spectrogram_mel_log(self, *, nfft: int = 2048, window: int = 512, stride: int = 256, mels: int = 256) -> tf.Tensor:
        spectrogram = tfio.audio.spectrogram(self._audio_tensor, nfft=nfft, window=window, stride=stride)
        mel_spectrogram = tfio.audio.melscale(spectrogram, rate=self._sample_rate, mels=mels, fmin=0, fmax=self._sample_rate // 2)
        db_scale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=50)
        db_scale_mel_spectrogram = tf.expand_dims(db_scale_mel_spectrogram, axis=2)
        return db_scale_mel_spectrogram