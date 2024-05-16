import tensorflow as tf
import tensorflow_io as tfio

from src.preprocess.audio_metafiles import AudioMetaInfo
from src.preprocess.split import SplitSet
from src.hyperparams.paths_info import PathsInfo


class Preprocess:
    SAMPLE_RATE = 48_000
    MAX_CLIENT_ID_AMOUNT = 1000
    MIN_CLIP_DURATION_MS = 4000
    SET_SIZE = 3000

    @staticmethod
    def add_zeros(wav, sample_rate):
        time_probes = wav.shape[0]
        missing_probes_one_side = int((Preprocess.MIN_CLIP_DURATION_MS / 1000 * sample_rate - time_probes) // 2)
        padded_tensor = tf.pad(wav.numpy(), [[missing_probes_one_side, missing_probes_one_side]])
        return tf.convert_to_tensor(padded_tensor, dtype=tf.float32)

    @staticmethod
    def cut_wav(wav, sample_rate):
        time_probes = wav.shape[0]
        # clip_dur_in_sec = time_probes / sample_rate
        overlap = int((time_probes - (Preprocess.MIN_CLIP_DURATION_MS / 1000) * sample_rate) / 2)
        cut_clip = wav[overlap:(time_probes - overlap)]
        return tf.convert_to_tensor(cut_clip, dtype=tf.float32)

    @staticmethod
    def load_audio(filename, fin_sam_rate=16_000):
        file_content = tf.io.read_file(filename)
        audio = tfio.audio.decode_mp3(file_content)
        audio = tf.squeeze(audio, axis=-1)
        # sample_rate = tf.cast(sample_rate, dtype=tf.int32)
        # wav = librosa.resample(wav.numpy(), orig_sr=sample_rate.numpy(), target_sr=fin_sam_rate)
        # return wav
        # return tf.convert_to_tensor(wav, dtype=tf.float32), sample_rate.numpy()
        return tf.convert_to_tensor(audio, dtype=tf.float32)

    @staticmethod
    def load_and_align_probes(file_path):
        audio = Preprocess.load_audio(file_path)
        expected_probes = int((Preprocess.MIN_CLIP_DURATION_MS / 1000) * Preprocess.SAMPLE_RATE)
        # print(expected_probes)
        current_probes = audio.shape[0]
        # print(current_probes)
        if expected_probes > current_probes:
            # print("Add zeros")
            return Preprocess.add_zeros(audio, Preprocess.SAMPLE_RATE)
        elif expected_probes < current_probes:
            # print("Cut wav")
            return Preprocess.cut_wav(audio, Preprocess.SAMPLE_RATE)
        return tf.convert_to_tensor(audio, dtype=tf.float32)

    @staticmethod
    def preprocess():
        train_dataset = None
        val_dataset = None
        test_dataset = None
        for lang in PathsInfo.get_languages():
            print(f"PROCESSED LANGUAGES: {lang}")
            audio_meta_info = AudioMetaInfo(
                max_client_id_amount=Preprocess.MAX_CLIENT_ID_AMOUNT,
                min_clip_duration_ms=Preprocess.MIN_CLIP_DURATION_MS,
                set_size=Preprocess.SET_SIZE,
                lang=lang
            )

            print(f"---AUDIO META INFO DONE---")

            df_men = audio_meta_info.get_df_men()
            df_women = audio_meta_info.get_df_women()

            df_filenames = SplitSet(
                df=df_men,
                max_client_id_amount=Preprocess.MAX_CLIENT_ID_AMOUNT,
                min_clip_duration_ms=Preprocess.MIN_CLIP_DURATION_MS,
                set_size=Preprocess.SET_SIZE,
                lang=lang
            ).get_filenames()

            print(f"---MEN SET SPLIT DONE---")

            train_dataset = tf.data.Dataset.from_tensor_slices(
                df_filenames.get('train').apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()
            )
            val_dataset = tf.data.Dataset.from_tensor_slices(
                df_filenames.get('val').apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()
            )
            test_dataset = tf.data.Dataset.from_tensor_slices(
                df_filenames.get('test').apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()
            )

            print(f"---DATASET MEN DONE DONE---")

            # TODO AUDIO AUGUMENT CLASS, REST OF DATA PIPELINE

            # print(train_dataset.as_numpy_iterator().next())
            # train_filenames, val_filenames, test_filenames = SplitSet(
            #     df=df_women,
            #     max_client_id_amount=Preprocess.MAX_CLIENT_ID_AMOUNT,
            #     min_clip_duration_ms=Preprocess.MIN_CLIP_DURATION_MS,
            #     set_size=Preprocess.SET_SIZE,
            #     lang=lang
            # ).get_filenames()
            #
            #
            # train_dataset = tf.data.Dataset.from_tensor_slices(
            #     train_filenames.apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()
            # )
            # val_dataset = tf.data.Dataset.from_tensor_slices(
            #     val_filenames.apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()
            # )
            # test_dataset = tf.data.Dataset.from_tensor_slices(
            #     test_dataset.apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()
            # )


Preprocess.preprocess()