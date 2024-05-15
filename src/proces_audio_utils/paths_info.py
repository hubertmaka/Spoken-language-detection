import os
import src.proces_audio_utils.exceptions as ex


class MetaPathsInfo(type):
    def __init__(cls, name, bases, attrs) -> None:
        super(MetaPathsInfo, cls).__init__(name, bases, attrs)
        if hasattr(cls, 'initialize'):
            cls.initialize()


class PathsInfo(metaclass=MetaPathsInfo):
    VALIDATED_FILENAME = 'validated.tsv'
    CLIP_DURATIONS_FILENAME = 'clip_durations.tsv'
    LANG_DIR_PATH: str = os.path.join("..", "..", "..", "languages")
    LANG_DIRS: list[str] = []
    VALIDATED_INFO_FILES_PATHS: list[str] = []
    DURATION_INFO_FILES_PATHS: list[str] = []

    @classmethod
    def get_lang_path(cls) -> list[str]:
        return os.listdir(cls.LANG_DIR_PATH)

    @classmethod
    def get_languages(cls) -> list[str]:
        return [os.path.join(cls.LANG_DIR_PATH, lang) for lang in cls.get_lang_path()]

    @classmethod
    def _update_lang_dirs(cls):
        cls.LANG_DIRS = cls.get_languages()

    @classmethod
    def _update_validated_info_files(cls):
        cls.VALIDATED_INFO_FILES_PATHS = [os.path.join(lang, cls.VALIDATED_FILENAME) for lang in cls.LANG_DIRS]

    @classmethod
    def _update_duration_info_files(cls):
        cls.DURATION_INFO_FILES_PATHS = [os.path.join(lang, cls.CLIP_DURATIONS_FILENAME) for lang in cls.LANG_DIRS]

    @classmethod
    def initialize(cls) -> None:
        cls._update_lang_dirs()
        cls._update_validated_info_files()
        cls._update_duration_info_files()


p = PathsInfo()
print(p.VALIDATED_INFO_FILES_PATHS)
print(p.DURATION_INFO_FILES_PATHS)