import os
import subprocess


class MetaPathsInfo(type):
    def __init__(cls, name, bases, attrs) -> None:
        super(MetaPathsInfo, cls).__init__(name, bases, attrs)
        if hasattr(cls, 'initialize'):
            cls.initialize()


class PathsInfo(metaclass=MetaPathsInfo):
    VALIDATED_FILENAME = 'validated.tsv'
    CLIP_DURATIONS_FILENAME = 'clip_durations.tsv'
    LANG_DIR_PATH: str
    LANG_DIRS: dict[str, str] = {}
    VALIDATED_INFO_FILES_PATHS: dict[str, str] = []
    DURATION_INFO_FILES_PATHS: dict[str, str] = []

    @classmethod
    def get_lang_dir_path(cls) -> None:
        result = subprocess.run(['find', '../../..', '-type', 'd', '-name', 'languages'],
                                capture_output=True, text=True)
        languages = result.stdout.strip()
        cls.LANG_DIR_PATH = languages

    @classmethod
    def get_languages(cls) -> list[str]:
        return os.listdir(cls.LANG_DIR_PATH)

    @classmethod
    def get_languages_paths(cls) -> list[str]:
        return [os.path.join(cls.LANG_DIR_PATH, lang) for lang in cls.get_languages()]

    @classmethod
    def _update_lang_dirs(cls):
        cls.LANG_DIRS = {lang: path for lang, path in zip(cls.get_languages(), cls.get_languages_paths())}

    @classmethod
    def _update_validated_info_files(cls):
        cls.VALIDATED_INFO_FILES_PATHS = {
            lang: os.path.join(path, cls.VALIDATED_FILENAME)
            for lang, path in cls.LANG_DIRS.items()
        }

    @classmethod
    def _update_duration_info_files(cls):
        cls.DURATION_INFO_FILES_PATHS = {
            lang: os.path.join(path, cls.CLIP_DURATIONS_FILENAME)
            for lang, path in cls.LANG_DIRS.items()
        }

    @classmethod
    def initialize(cls) -> None:
        cls.get_lang_dir_path()
        cls._update_lang_dirs()
        cls._update_validated_info_files()
        cls._update_duration_info_files()
