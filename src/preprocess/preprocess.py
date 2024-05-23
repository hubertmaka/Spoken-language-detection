from typing import Any

import random

import src.preprocess.exceptions as ex
from src.preprocess.meta_info import AudioMetaInfo
from src.preprocess.split import SplitSet
from src.hyperparams.paths_info import PathsInfo

from src.hyperparams.set_hparams import SetHyperparameters


class Preprocess:
    ORIGIN_SAMPLE_RATE: int = 48_000
    SAMPLE_RATE: int = 16_000
    MAX_CLIENT_ID_AMOUNT: int = 1000
    MIN_CLIP_DURATION_MS: int = 4000
    SET_SIZE: int
    ONE_LANG_SET_SIZE: int
    ONE_LANG_GENDER_SET_SIZE: int
    SET_HPARAMS: SetHyperparameters
    BATCH_SIZE: int = 32
    TRAIN_FILENAMES: list[tuple[str, int]] = []
    VAL_FILENAMES: list[tuple[str, int]] = []
    TEST_FILENAMES: list[tuple[str, int]] = []
    # train_dataset = tf.data.Dataset.from_tensor_slices(([], []))
    # val_dataset = tf.data.Dataset.from_tensor_slices(([], []))
    # test_dataset = tf.data.Dataset.from_tensor_slices(([], []))
    MIN_TRAIN_SET_FILL = 100
    LANGUAGES = PathsInfo.get_languages()
    LANGUAGES_TO_INDEX = {lang: idx for idx, lang in enumerate(LANGUAGES)}

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
        cls.SET_SIZE = set_size
        cls.ONE_LANG_SET_SIZE = (cls.SET_SIZE // len(PathsInfo.get_languages()))
        cls.ONE_LANG_GENDER_SET_SIZE = cls.ONE_LANG_SET_SIZE // 2 if cls.SET_SIZE % 2 == 0 else cls.ONE_LANG_SET_SIZE // 2 - 1
        cls.SET_HPARAMS = SetHyperparameters(cls.ONE_LANG_SET_SIZE)

    @classmethod
    def set_batch_size(cls, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        cls.BATCH_SIZE = batch_size

    @classmethod
    def _split_filenames(cls) -> None:
        for lang in PathsInfo.get_languages():
            men_size = cls.ONE_LANG_GENDER_SET_SIZE
            women_size = cls.ONE_LANG_GENDER_SET_SIZE
            print(f"PROCESSED LANGUAGES: {lang}")
            audio_meta_info = AudioMetaInfo(
                max_client_id_amount=cls.MAX_CLIENT_ID_AMOUNT,
                min_clip_duration_ms=cls.MIN_CLIP_DURATION_MS,
                set_size=cls.ONE_LANG_GENDER_SET_SIZE,
                lang=lang
            )

            print(f"---AUDIO META INFO DONE {lang}---")

            df_men = audio_meta_info.get_df_men()
            df_women = audio_meta_info.get_df_women()

            if len(df_women) <= cls.SET_HPARAMS.test_size // 2 + cls.SET_HPARAMS.val_size // 2 + cls.MIN_TRAIN_SET_FILL:
                missing_women_probes = cls.SET_HPARAMS.test_size // 2 + cls.SET_HPARAMS.val_size // 2 + cls.MIN_TRAIN_SET_FILL - len(df_women)
                men_size = cls.ONE_LANG_GENDER_SET_SIZE + missing_women_probes
                women_size = cls.ONE_LANG_GENDER_SET_SIZE - missing_women_probes
                print("MISSING PROBES")

            for num, gender_df in enumerate([df_men, df_women]):
                tuned_set_size = SetHyperparameters(men_size) if num == 0 else SetHyperparameters(women_size)

                df_filenames = SplitSet(
                    df=gender_df,
                    max_client_id_amount=cls.MAX_CLIENT_ID_AMOUNT,
                    min_clip_duration_ms=cls.MIN_CLIP_DURATION_MS,
                    set_size=(men_size if num == 0 else women_size),
                    lang=lang
                ).get_filenames()


                print(f"---SET SPLIT DONE {num} {lang}---")
                # TODO: tutuaj brać dany gender sprawdzać czy ilość plików jest równna danemu zbiorowi treningowemu (val i test przekazywać dalej)
                # TODO: nastepnie jeżeli dana płeć jest mniejsza od danego zbioru to wybrać tyle losowych próbek żeby dopełnić
                # TODO: potem w datasecie zaugumentować. Dodawać do listy ścieżek do plików i wziąć zrobić dataset jako from tensor slices


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

                cls.TEST_FILENAMES.extend([(path, cls.LANGUAGES_TO_INDEX.get(lang)) for path in df_filenames.get('test')])
                cls.VAL_FILENAMES.extend([(path, cls.LANGUAGES_TO_INDEX.get(lang)) for path in df_filenames.get('val')])
                cls.TRAIN_FILENAMES.extend([(path, cls.LANGUAGES_TO_INDEX.get(lang)) for path in df_filenames.get('train')])

                missing_probes = tuned_set_size.train_size - len(df_filenames.get('train'))
                # print(f"Missing probes: {missing_probes}")
                cls.TRAIN_FILENAMES.extend([(path, cls.LANGUAGES_TO_INDEX.get(lang)) for path in df_filenames.get('train').sample(missing_probes, replace=True)])

                # print("Tuned train size:", tuned_set_size.train_size)
                # print("Tuned val size:", tuned_set_size.val_size)
                # print("Tuned test size:", tuned_set_size.test_size)
                # print("--------------------------")
                # print("Test size: ", len(df_filenames.get('test')))
                # print("Val size: ", len(df_filenames.get('val')))
                # print("Train size: ", len(df_filenames.get('train')))
                # print("--------------------------")
                # print("--------------------------")

    @classmethod
    def _shuffle_filenames(cls) -> None:
        random.shuffle(cls.TEST_FILENAMES)
        random.shuffle(cls.VAL_FILENAMES)
        random.shuffle(cls.TRAIN_FILENAMES)

    @classmethod
    def initialize(cls, **kwargs: Any) -> None:
        attributes: dict[str, Any] = {key: val for key, val in kwargs.items()}
        cls.set_origin_sample_rate(attributes['origin_sample_rate'])
        cls.set_final_sample_rate(attributes['final_sample_rate'])
        cls.set_max_client_id_amount(attributes['max_client_id_amount'])
        cls.set_min_clip_duration_ms(attributes['min_clip_duration_ms'])
        cls.set_set_size(attributes['set_size'])
        cls.set_batch_size(attributes['batch_size'])

    @classmethod
    def preprocess_probes(cls) -> None:
        cls._split_filenames()
        cls._shuffle_filenames()


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



#
