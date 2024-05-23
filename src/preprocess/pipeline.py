import random

from src.hyperparams.paths_info import PathsInfo
from src.preprocess.preprocess import Preprocess
import tensorflow as tf
import tensorflow_io as tfio

from src.preprocess.audio import ProcessAudio


class Pipeline:
    languages = PathsInfo.get_languages()
    language_to_index = {lang: idx for idx, lang in enumerate(languages)}

    @staticmethod
    @tf.function
    def one_hot_encode_language(lang):
        lang_index = Pipeline.language_to_index[lang]
        one_hot = tf.one_hot(lang_index, len(Pipeline.languages))
        return one_hot

    @staticmethod
    def set_shapes(audio, label, audio_shape, label_shape):
        audio.set_shape(audio_shape)
        label.set_shape(label_shape)
        return audio, label

    @staticmethod
    def add_zeros(audio: tf.Tensor, sample_rate) -> tf.Tensor:
        time_probes = tf.cast(tf.shape(audio)[0], dtype=tf.int32)
        min_clip_duration_probes = tf.cast((Preprocess.MIN_CLIP_DURATION_MS / 1000) * sample_rate, dtype=tf.int32)
        missing_probes_one_side = tf.cast((min_clip_duration_probes - time_probes) / 2, dtype=tf.int32)
        padded_tensor = tf.pad(audio, [[missing_probes_one_side, missing_probes_one_side]])
        return padded_tensor

    @staticmethod
    def cut_audio(audio: tf.Tensor, sample_rate: int) -> tf.Tensor:
        time_probes = tf.cast(tf.shape(audio)[0], dtype=tf.int32)
        min_clip_duration_probes = tf.cast((Preprocess.MIN_CLIP_DURATION_MS / 1000) * sample_rate, dtype=tf.int32)
        overlap = tf.cast((time_probes - min_clip_duration_probes) / 2, dtype=tf.int32)
        cut_clip = audio[overlap:(time_probes - overlap)]
        return cut_clip

    @staticmethod
    def load_resample_audio(filename: str) -> tf.Tensor:
        file_content = tf.io.read_file(filename)
        audio = tfio.audio.decode_mp3(file_content)
        audio = tf.squeeze(audio, axis=-1)
        audio = Pipeline.resample_audio(audio, Preprocess.ORIGIN_SAMPLE_RATE, Preprocess.SAMPLE_RATE)
        return audio

    @staticmethod
    def resample_audio(audio: tf.Tensor, curr_sr: int, fin_sr: int) -> tf.Tensor:
        curr_sr = tf.cast(curr_sr, tf.int64)
        return tfio.audio.resample(audio, rate_in=curr_sr, rate_out=fin_sr)

    @staticmethod
    def align_probes(audio: tf.Tensor, sample_rate: int) -> tf.Tensor:
        expected_probes = tf.cast((Preprocess.MIN_CLIP_DURATION_MS / 1000) * Preprocess.SAMPLE_RATE, dtype=tf.int32)
        current_probes = tf.cast(tf.shape(audio)[0], dtype=tf.int32)
        if expected_probes > current_probes:
            print("ADD ZEROS")
            audio = Pipeline.add_zeros(audio, sample_rate)
        elif expected_probes < current_probes:
            print("CUT AUDIO")
            audio = Pipeline.cut_audio(audio, sample_rate)
        return audio

    @staticmethod
    def load_and_align_probes(file_path: str) -> tf.Tensor:
        audio = Pipeline.load_resample_audio(file_path)
        audio = Pipeline.align_probes(audio, Preprocess.SAMPLE_RATE)
        return audio

    @staticmethod
    def augment_audio(audio: tf.Tensor) -> tf.Tensor:
        sample_rate = Preprocess.SAMPLE_RATE
        audio = ProcessAudio(audio, sample_rate).add_noise_rand(min_level=0.01, max_level=0.06)
        audio = ProcessAudio(audio, sample_rate).change_amplitude_rand(min_increase=0.7, max_increase=1.3)

        if 25 <= random.randint(1, 70) <= 45:
            audio, sample_rate = random.choice([
                ProcessAudio(audio, sample_rate).time_masking_rand(
                    min_mask=500, max_mask=2300
                ),
                ProcessAudio(audio, sample_rate).change_pitch_rand(
                    sample_rate=sample_rate, min_shift=-2, max_shift=2
                ),
                ProcessAudio(audio, sample_rate).time_shift_rand(
                    min_shift=-300, max_shift=300
                ),
            ])
        print("AUGUMENT")
        return audio

    @staticmethod
    def one_hot_encode(lang_index):
        one_hot = tf.one_hot(lang_index, len(Pipeline.languages), dtype=tf.int32)
        return one_hot

    @staticmethod
    def create_pipeline(dataset: list[tuple[str, int]], is_train: bool = False) -> tf.data.Dataset:
        train_filenames = [x[0] for x in dataset]
        train_labels = [x[1] for x in dataset]
        dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
        # print(dataset.as_numpy_iterator().next())
        dataset = dataset.map(lambda filename, label: (filename, Pipeline.one_hot_encode(label)))
        # print(dataset.as_numpy_iterator().next())
        dataset = dataset.map(lambda filename, label: (Pipeline.load_and_align_probes(filename), label))
        # print(dataset.as_numpy_iterator().next())
        if is_train:
            dataset = dataset.map(lambda audio, label: (Pipeline.augment_audio(audio), label))
            # print(dataset.as_numpy_iterator().next())
        dataset = dataset.map(lambda audio, label: (ProcessAudio(audio, Preprocess.SAMPLE_RATE).normalize_audio(), label))
        # print(dataset.as_numpy_iterator().next())
        dataset = dataset.map(lambda audio, label: (ProcessAudio(audio, Preprocess.SAMPLE_RATE).create_spectrogram(), label))
        # print(dataset.as_numpy_iterator().next())
        # JEŻELI NIE BĘDZIE WIDZIAŁ TENSORFLOW KSZTAŁTÓW TO TO JEST PO TO ZEBY JE USTAWIĆ
        sample, label = dataset.as_numpy_iterator().next()
        dataset = dataset.map(lambda audio, label: Pipeline.set_shapes(audio, label, sample.shape, label.shape))
        dataset = dataset.cache()
        # print(dataset.as_numpy_iterator().next())
        # dataset = dataset.shuffle(buffer_size=5_000)
        # print(dataset.as_numpy_iterator().next())
        dataset = dataset.batch(batch_size=Preprocess.BATCH_SIZE, drop_remainder=True)
        # print(dataset.as_numpy_iterator().next())
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        # print(dataset.as_numpy_iterator().next())
        return dataset




# ORIGIN_SAMPLE_RATE = 48_000
# FINAL_SAMPLE_RATE = 16_000
# MAX_CLIENT_ID_AMOUNT = 3000
# MIN_CLIP_DURATION_MS = 2000
# SET_SIZE = 1_000
# BATCH_SIZE = 32
#
# Preprocess.initialize(
#     batch_size=BATCH_SIZE,
#     max_client_id_amount=MAX_CLIENT_ID_AMOUNT,
#     min_clip_duration_ms=MIN_CLIP_DURATION_MS,
#     set_size=SET_SIZE,
#     origin_sample_rate=ORIGIN_SAMPLE_RATE,
#     final_sample_rate=FINAL_SAMPLE_RATE,
# )
# Preprocess.preprocess_probes()
#
# Pipeline.create_pipeline(Preprocess.TRAIN_FILENAMES, is_train=True)
