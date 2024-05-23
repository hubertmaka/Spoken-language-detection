from src.hyperparams.paths_info import PathsInfo
from src.preprocess.preprocess import Preprocess
import tensorflow as tf
import tensorflow_io as tfio


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
        time_probes = audio.shape[0]
        missing_probes_one_side = int((Preprocess.MIN_CLIP_DURATION_MS / 1000 * sample_rate - time_probes) / 2)
        padded_tensor = tf.pad(audio.numpy(), [[missing_probes_one_side, missing_probes_one_side]])
        return tf.convert_to_tensor(padded_tensor, dtype=tf.float32)

    @staticmethod
    def cut_audio(audio: tf.Tensor, sample_rate: int) -> tf.Tensor:
        time_probes = audio.shape[0]
        overlap = int((time_probes - (Preprocess.MIN_CLIP_DURATION_MS / 1000) * sample_rate) / 2)
        cut_clip = audio[overlap:(time_probes - overlap)]
        return tf.convert_to_tensor(cut_clip, dtype=tf.float32)

    @staticmethod
    def load_resample_audio(filename: str) -> tf.Tensor:
        file_content = tf.io.read_file(filename)
        audio = tfio.audio.decode_mp3(file_content)
        audio = tf.squeeze(audio, axis=-1)
        audio = Pipeline.resample_audio(audio, Preprocess.ORIGIN_SAMPLE_RATE, Preprocess.SAMPLE_RATE)
        return tf.convert_to_tensor(audio, dtype=tf.float32)

    @staticmethod
    @tf.function(reduce_retracing=True)
    def resample_audio(audio: tf.Tensor, curr_sr: int, fin_sr: int) -> tf.Tensor:
        curr_sr = tf.cast(curr_sr, tf.int64)
        return tfio.audio.resample(audio, rate_in=curr_sr, rate_out=fin_sr)

    @staticmethod
    def align_probes(audio: tf.Tensor, sample_rate: int) -> tf.Tensor:
        expected_probes = int((Preprocess.MIN_CLIP_DURATION_MS / 1000) * Preprocess.SAMPLE_RATE)
        current_probes = audio.shape[0]
        if expected_probes > current_probes:
            audio = Pipeline.add_zeros(audio, sample_rate)
        elif expected_probes < current_probes:
            audio = Pipeline.cut_audio(audio, sample_rate)
        return tf.convert_to_tensor(audio, dtype=tf.float32)

    @staticmethod
    def load_and_align_probes(file_path: str) -> tf.Tensor:
        audio = Pipeline.load_resample_audio(file_path)
        expected_probes = int((Preprocess.MIN_CLIP_DURATION_MS / 1000) * Preprocess.SAMPLE_RATE)
        current_probes = audio.shape[0]
        if expected_probes > current_probes:
            audio = Pipeline.add_zeros(audio, Preprocess.SAMPLE_RATE)
        elif expected_probes < current_probes:
            audio = Pipeline.cut_audio(audio, Preprocess.SAMPLE_RATE)
        return tf.convert_to_tensor(audio, dtype=tf.float32)

    @staticmethod
    def create_pipeline(train: list[tuple[str, int]], val: list[tuple[str, int]], test: list[tuple[str, int]]):
        pass

    # @staticmethod
    # def process_random_samples(dataset: tf.data.Dataset, num_samples_to_process: int) -> list[
    #     tuple[tf.Tensor, tf.Tensor]]:
    #     processed_samples = []
    #
    #     shuffled_dataset = dataset.shuffle(buffer_size=10)
    #     samples = shuffled_dataset.take(num_samples_to_process)
    #     for audio, label in samples:
    #         audio = ProcessAudio(audio, Preprocess.SAMPLE_RATE).add_noise_rand()
    #         audio = ProcessAudio(audio, Preprocess.SAMPLE_RATE).change_amplitude_rand()
    #         audio = random.choice([
    #             ProcessAudio(audio, Preprocess.SAMPLE_RATE).time_masking_rand(),
    #         ])
    #         processed_samples.append((audio, label))
    #     return processed_samples