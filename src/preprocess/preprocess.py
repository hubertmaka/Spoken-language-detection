from typing import Any, List, Tuple

import tensorflow as tf
import tensorflow_io as tfio
import random

import src.preprocess.exceptions as ex
from src.preprocess.audio_metafiles import AudioMetaInfo
from src.preprocess.split import SplitSet
from src.hyperparams.paths_info import PathsInfo
from src.preprocess.process_audio import ProcessAudio
from src.hyperparams.set_hyperparameters import SetHyperparameters


class Preprocess:
    ORIGIN_SAMPLE_RATE: int = 48_000
    SAMPLE_RATE: int = 16_000
    MAX_CLIENT_ID_AMOUNT: int = 1000
    MIN_CLIP_DURATION_MS: int = 4000
    SET_SIZE: int
    ONE_LANG_SET_SIZE: int
    SET_HPARAMS: SetHyperparameters
    BATCH_SIZE: int = 32
    TRAIN_FILENAMES: list[str] = []
    VAL_FILENAMES: list[str] = []
    TEST_FILENAMES: list[str] = []
    train_dataset = tf.data.Dataset.from_tensor_slices(([], []))
    val_dataset = tf.data.Dataset.from_tensor_slices(([], []))
    test_dataset = tf.data.Dataset.from_tensor_slices(([], []))
    MIN_TRAIN_SET_FILL = 10

    @classmethod
    def set_origin_sample_rate(cls, rate: int) -> None:
        if rate <= 0:
            raise ValueError("Rate must be positive")
        cls.ORIGIN_SAMPLE_RATE = rate

    @classmethod
    def set_final_sample_rate(cls, rate: int) -> None:
        if rate <= 0:
            raise ValueError("Rate must be positive")
        cls.SAMPLE_RATE = rate

    @classmethod
    def set_max_client_id_amount(cls, amount: int) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        cls.MAX_CLIENT_ID_AMOUNT = amount

    @classmethod
    def set_min_clip_duration_ms(cls, duration_ms: int) -> None:
        if duration_ms <= 0:
            raise ValueError("Duration must be positive")
        cls.MIN_CLIP_DURATION_MS = duration_ms

    @classmethod
    def set_set_size(cls, set_size: int) -> None:
        if set_size <= 0:
            raise ValueError("Set size must be positive")
        # print(type(cls.SET_SIZE))
        cls.SET_SIZE = set_size
        cls.ONE_LANG_SET_SIZE = (cls.SET_SIZE // len(PathsInfo.get_languages()))
        cls.ONE_LANG_GENDER_SET_SIZE = cls.ONE_LANG_SET_SIZE // 2 if cls.SET_SIZE % 2 == 0 else cls.ONE_LANG_SET_SIZE // 2 - 1
        cls.SET_HPARAMS = SetHyperparameters(cls.ONE_LANG_SET_SIZE)
        print(type(cls.SET_SIZE))

    @classmethod
    def set_batch_size(cls, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        cls.BATCH_SIZE = batch_size

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
        audio = Preprocess.resample_audio(audio, Preprocess.ORIGIN_SAMPLE_RATE, Preprocess.SAMPLE_RATE)
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
            audio = Preprocess.add_zeros(audio, sample_rate)
        elif expected_probes < current_probes:
            audio = Preprocess.cut_audio(audio, sample_rate)
        return tf.convert_to_tensor(audio, dtype=tf.float32)

    @staticmethod
    def load_and_align_probes(file_path: str) -> tf.Tensor:
        audio = Preprocess.load_resample_audio(file_path)
        expected_probes = int((Preprocess.MIN_CLIP_DURATION_MS / 1000) * Preprocess.SAMPLE_RATE)
        current_probes = audio.shape[0]
        if expected_probes > current_probes:
            audio = Preprocess.add_zeros(audio, Preprocess.SAMPLE_RATE)
        elif expected_probes < current_probes:
            audio = Preprocess.cut_audio(audio, Preprocess.SAMPLE_RATE)
        return tf.convert_to_tensor(audio, dtype=tf.float32)

    @staticmethod
    def process_random_samples(dataset: tf.data.Dataset, num_samples_to_process: int) -> list[tuple[tf.Tensor, tf.Tensor]]:
        processed_samples = []

        shuffled_dataset = dataset.shuffle(buffer_size=10)
        samples = shuffled_dataset.take(num_samples_to_process)
        for audio, label in samples:
            audio = ProcessAudio(audio, Preprocess.SAMPLE_RATE).add_noise_rand()
            audio = ProcessAudio(audio, Preprocess.SAMPLE_RATE).change_amplitude_rand()
            audio = random.choice([
                ProcessAudio(audio, Preprocess.SAMPLE_RATE).time_masking_rand(),
            ])
            processed_samples.append((audio, label))
        return processed_samples

    @staticmethod
    def preprocess() -> None:

        languages = PathsInfo.get_languages()
        language_to_index = {lang: idx for idx, lang in enumerate(languages)}

        @tf.function
        def one_hot_encode_language(lang):
            lang_index = language_to_index[lang]
            one_hot = tf.one_hot(lang_index, len(languages))
            return one_hot

        def set_shapes(audio, label, audio_shape, label_shape):
            audio.set_shape(audio_shape)
            label.set_shape(label_shape)
            return audio, label


        need_fill = False

        for lang in PathsInfo.get_languages():
            men_size = Preprocess.ONE_LANG_GENDER_SET_SIZE
            women_size = Preprocess.ONE_LANG_GENDER_SET_SIZE
            print(f"PROCESSED LANGUAGES: {lang}")
            audio_meta_info = AudioMetaInfo(
                max_client_id_amount=Preprocess.MAX_CLIENT_ID_AMOUNT,
                min_clip_duration_ms=Preprocess.MIN_CLIP_DURATION_MS,
                set_size=Preprocess.ONE_LANG_GENDER_SET_SIZE,
                lang=lang
            )

            print(f"---AUDIO META INFO DONE {lang}---")

            df_men = audio_meta_info.get_df_men()
            df_women = audio_meta_info.get_df_women()

            if len(df_women) <= Preprocess.SET_HPARAMS.test_size // 2 + Preprocess.SET_HPARAMS.val_size // 2 + Preprocess.MIN_TRAIN_SET_FILL:
                missing_women_probes = Preprocess.SET_HPARAMS.test_size // 2 + Preprocess.SET_HPARAMS.val_size // 2 + Preprocess.SET_HPARAMS.train_size // 2 + Preprocess.MIN_TRAIN_SET_FILL - len(df_women)
                men_size = Preprocess.ONE_LANG_GENDER_SET_SIZE + missing_women_probes
                women_size = Preprocess.ONE_LANG_GENDER_SET_SIZE - missing_women_probes
                # print(missing_women_probes)
                # print(Preprocess.ONE_LANG_GENDER_SET_SIZE + missing_women_probes)
                # print(Preprocess.ONE_LANG_GENDER_SET_SIZE - missing_women_probes)
                need_fill = True


            # del audio_meta_info

            for num, gender_df in enumerate([df_men, df_women]):
                tuned_set_size = SetHyperparameters(men_size) if num == 0 else SetHyperparameters(women_size)
                if not need_fill:
                    # tuned_set_size = SetHyperparameters(Preprocess.ONE_LANG_GENDER_SET_SIZE)

                    df_filenames = SplitSet(
                        df=gender_df,
                        max_client_id_amount=Preprocess.MAX_CLIENT_ID_AMOUNT,
                        min_clip_duration_ms=Preprocess.MIN_CLIP_DURATION_MS,
                        set_size=(men_size if num == 0 else women_size),
                        lang=lang
                    ).get_filenames()
                else:
                    # tuned_set_size = SetHyperparameters(men_size) if num == 0 else SetHyperparameters(women_size)
                    print("TUNING SET SIZE")
                    df_filenames = SplitSet(
                        df=gender_df,
                        max_client_id_amount=Preprocess.MAX_CLIENT_ID_AMOUNT,
                        min_clip_duration_ms=Preprocess.MIN_CLIP_DURATION_MS,
                        set_size=(men_size if num == 0 else women_size),
                        lang=lang
                    ).get_filenames()

                print(f"---SET SPLIT DONE {num} {lang}---")
                # TODO: tutuaj brać dany gender sprawdzać czy ilość plików jest równna danemu zbiorowi treningowemu (val i test przekazywać dalej)
                # TODO: nastepnie jeżeli dana płeć jest mniejsza od danego zbioru to wybrać tyle losowych próbek żeby dopełnić
                # TODO: potem w datasecie zaugumentować. Dodawać do listy ścieżek do plików i wziąć zrobić dataset jako from tensor slices
                # print(len(df_filenames.get('train')))
                # print(len(df_filenames.get('val')))
                # print()



                if len(df_filenames.get('test')) < (tuned_set_size.test_size // 2):
                    raise ex.NotEnoughProbesInSetError(
                        f"test {lang} {num}",
                        tuned_set_size.test_size // 2 - len(df_filenames.get('test'))
                    )
                if len(df_filenames.get('val')) < (tuned_set_size.val_size // 2):
                    raise ex.NotEnoughProbesInSetError(
                        f"val {lang} {num}",
                        tuned_set_size.val_size // 2 - len(df_filenames.get('val'))
                    )

                Preprocess.TEST_FILENAMES.extend(df_filenames.get('test'))
                Preprocess.VAL_FILENAMES.extend(df_filenames.get('val'))
                Preprocess.TRAIN_FILENAMES.extend(df_filenames.get('train'))
                print(f"train size: {tuned_set_size.train_size}")
                print(f"len train: {len(df_filenames.get('train'))}")

                SetHyperparameters(men_size)

                missing_probes = tuned_set_size.train_size - len(df_filenames.get('train'))
                print(f"Missing probes: {missing_probes}")
                Preprocess.TRAIN_FILENAMES.extend(df_filenames.get('train').sample(missing_probes, replace=True))

                print(len(Preprocess.TEST_FILENAMES))
                print(len(Preprocess.VAL_FILENAMES))
                print(len(Preprocess.TRAIN_FILENAMES))

                print("--------------------------")
                print(len(df_filenames.get('test')))
                print(len(df_filenames.get('val')))
                print(len(df_filenames.get('train')))
                print("--------------------------")
                print("--------------------------")
                # train_filenames = df_filenames.get('train').apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()
                # val_filenames = df_filenames.get('val').apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()
                # test_filenames = df_filenames.get('test').apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()
                #
                # print(train_filenames)
                #
                # train_labels = [one_hot_encode_language(lang)] * len(train_filenames)
                # val_labels = [one_hot_encode_language(lang)] * len(val_filenames)
                # test_labels = [one_hot_encode_language(lang)] * len(test_filenames)
                #
                # train_dataset_tmp = tf.data.Dataset.zip((
                #     tf.data.Dataset.from_tensor_slices(train_filenames),
                #     tf.data.Dataset.from_tensor_slices(train_labels)
                # ))
                # print(train_dataset_tmp)
                # val_dataset_tmp = tf.data.Dataset.zip((
                #     tf.data.Dataset.from_tensor_slices(val_filenames),
                #     tf.data.Dataset.from_tensor_slices(val_labels)
                # ))
                # test_dataset_tmp = tf.data.Dataset.zip((
                #     tf.data.Dataset.from_tensor_slices(test_filenames),
                #     tf.data.Dataset.from_tensor_slices(test_labels)
                # ))
                # NEW START

                # train_filenames =
                # val_filenames =
                # test_filenames =
                #
                # print(train_filenames)
                #
                # train_labels =
                # val_labels =
                # test_labels =

                # train_dataset_tmp = tf.data.Dataset.zip((
                #     tf.data.Dataset.from_tensor_slices(df_filenames.get('train').apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()),
                #     tf.data.Dataset.from_tensor_slices([one_hot_encode_language(lang)] * len(df_filenames.get('train')))
                # ))
                # print(train_dataset_tmp)
                # val_dataset_tmp = tf.data.Dataset.zip((
                #     tf.data.Dataset.from_tensor_slices(df_filenames.get('val').apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()),
                #     tf.data.Dataset.from_tensor_slices([one_hot_encode_language(lang)] * len(df_filenames.get('val')))
                # ))
                # test_dataset_tmp = tf.data.Dataset.zip((
                #     tf.data.Dataset.from_tensor_slices(df_filenames.get('test').apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()),
                #     tf.data.Dataset.from_tensor_slices([one_hot_encode_language(lang)] * len(df_filenames.get('test')))
                # ))
                #
                # # NEW END
                #
                # print(f"---DATASET DONE {num} {lang}---")
                #
                # filled_samples = Preprocess.process_random_samples(
                #     train_dataset_tmp, (Preprocess.SET_HPARAMS.train_size // 2) - len(train_dataset_tmp)
                # )
                #
                # filled_audio_samples = tf.data.Dataset.from_tensor_slices([Preprocess.align_probes(tensor[0], Preprocess.SAMPLE_RATE) for tensor in filled_samples])
                # filled_labels = tf.data.Dataset.from_tensor_slices([tensor[1] for tensor in filled_samples])
                #
                # filled_samples = tf.data.Dataset.zip((filled_audio_samples, filled_labels))
                #
                # train_dataset_tmp = train_dataset_tmp.concatenate(filled_samples)
                #
                # Preprocess.train_dataset = Preprocess.train_dataset.concatenate(train_dataset_tmp)
                # Preprocess.val_dataset = Preprocess.val_dataset.concatenate(val_dataset_tmp)
                # Preprocess.test_dataset = Preprocess.test_dataset.concatenate(test_dataset_tmp)
                #
                # del df_filenames
                # del filled_samples
                # del train_dataset_tmp
                # del val_dataset_tmp
                # del test_dataset_tmp
            #
            # del df_men
            # del df_women
            # -------------------------------------

        # Preprocess.train_dataset = Preprocess.train_dataset.map(lambda audio, label: (ProcessAudio(audio, Preprocess.SAMPLE_RATE).normalize_audio(), label))
        # Preprocess.val_dataset = Preprocess.val_dataset.map(lambda audio, label: (ProcessAudio(audio, Preprocess.SAMPLE_RATE).normalize_audio(), label))
        # Preprocess.test_dataset = Preprocess.test_dataset.map(lambda audio, label: (ProcessAudio(audio, Preprocess.SAMPLE_RATE).normalize_audio(), label))
        #
        # Preprocess.train_dataset = Preprocess.train_dataset.map(lambda audio, label: (ProcessAudio(audio, Preprocess.SAMPLE_RATE).create_spectrogram(), label))
        # Preprocess.val_dataset = Preprocess.val_dataset.map(lambda audio, label: (ProcessAudio(audio, Preprocess.SAMPLE_RATE).create_spectrogram(), label))
        # Preprocess.test_dataset = Preprocess.test_dataset.map(lambda audio, label: (ProcessAudio(audio, Preprocess.SAMPLE_RATE).create_spectrogram(), label))
        #
        # sample, label = Preprocess.train_dataset.as_numpy_iterator().next()
        # Preprocess.train_dataset = Preprocess.train_dataset.map(lambda x, y: set_shapes(x, y, sample.shape, label.shape))
        # Preprocess.val_dataset = Preprocess.val_dataset.map(lambda x, y: set_shapes(x, y, sample.shape, label.shape))
        # Preprocess.test_dataset = Preprocess.test_dataset.map(lambda x, y: set_shapes(x, y, sample.shape, label.shape))
        #
        # # print("CACHING")
        # # train_dataset = train_dataset.cache()
        # # val_dataset = val_dataset.cache()
        # # test_dataset = test_dataset.cache()
        # print("SHUFFLE")
        # Preprocess.train_dataset = Preprocess.train_dataset.shuffle(buffer_size=(Preprocess.SET_HPARAMS.train_size // 3))
        # Preprocess.val_dataset = Preprocess.val_dataset.shuffle(buffer_size=(Preprocess.SET_HPARAMS.val_size // 3))
        # Preprocess.test_dataset = Preprocess.test_dataset.shuffle(buffer_size=(Preprocess.SET_HPARAMS.test_size // 3))
        #
        # print("BATCHING")
        # # train_dataset = train_dataset.batch(16, drop_remainder=True)
        # Preprocess.train_dataset = Preprocess.train_dataset.batch(Preprocess.BATCH_SIZE, drop_remainder=True)
        # Preprocess.val_dataset = Preprocess.val_dataset.batch(Preprocess.BATCH_SIZE, drop_remainder=True)
        # Preprocess.test_dataset = Preprocess.test_dataset.batch(Preprocess.BATCH_SIZE, drop_remainder=True)
        #
        # return train_dataset, val_dataset, test_dataset



ORIGIN_SAMPLE_RATE = 48_000
FINAL_SAMPLE_RATE = 16_000
MAX_CLIENT_ID_AMOUNT = 3000
MIN_CLIP_DURATION_MS = 2000
SET_SIZE = 30_000
BATCH_SIZE = 32
#%%
Preprocess.set_origin_sample_rate(ORIGIN_SAMPLE_RATE)
Preprocess.set_final_sample_rate(FINAL_SAMPLE_RATE)
Preprocess.set_max_client_id_amount(MAX_CLIENT_ID_AMOUNT)
Preprocess.set_min_clip_duration_ms(MIN_CLIP_DURATION_MS)
Preprocess.set_set_size(SET_SIZE)
Preprocess.set_batch_size(BATCH_SIZE)




Preprocess.preprocess()

print(len(Preprocess.TRAIN_FILENAMES))
print(len(Preprocess.VAL_FILENAMES))
print(len(Preprocess.TEST_FILENAMES))