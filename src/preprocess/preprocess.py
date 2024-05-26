import os.path
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

                cls.TRAIN_FILENAMES.extend([(path, cls.LANGUAGES_TO_INDEX.get(lang)) for path in df_filenames.get('train').sample(missing_probes, replace=True)])

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
    def save_set(cls, filename: str, set_to_save: list[tuple[str, int]]) -> None:
        directory = os.path.join(".", "data")
        if not os.path.exists(directory):
            os.makedirs(directory)

        filepath = os.path.join(directory, filename)
        with open(filepath, "w") as file_handler:
            for item in set_to_save:
                file_handler.write(f"{item[0]},{item[1]}\n")

    @classmethod
    def load_set(cls, filename: str) -> list[tuple[str, int]]:
        result = []
        try:
            with open(os.path.join(".", "data", filename), "r") as file_handler:
                for line in file_handler:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        result.append((parts[0], int(parts[1])))
        except FileNotFoundError:
            print(f"File not found: {filename}")
        except ValueError:
            print(f"Error while parsing the file: {filename}")
        return result

    @classmethod
    def preprocess_probes(cls) -> None:
        cls._split_filenames()
        cls._shuffle_filenames()
        cls.save_set("train", cls.TRAIN_FILENAMES)
        cls.save_set("val", cls.VAL_FILENAMES)
        cls.save_set("test", cls.TEST_FILENAMES)







