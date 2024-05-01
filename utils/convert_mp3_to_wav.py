import subprocess
import os


class ConvertMp3ToWav:
    def __init__(self, main_directory: str, mp3_directory: str, wav_directory: str):
        self._main_directory = main_directory
        self._wav_directory = wav_directory
        self._mp3_directory = mp3_directory
        self._available_languages_mp3: list[str] = self._load_path_to_mp3()
        self._available_clips: dict[str, list[str]] = self._load_available_clips()

    def _load_path_to_mp3(self) -> list[str]:
        try:
            return os.listdir(os.path.join(self._main_directory, self._mp3_directory))
        except FileNotFoundError:
            print("Directory not found. Creating new directory...")
            os.makedirs(os.path.join(self._main_directory, self._mp3_directory))
            return os.listdir(os.path.join(self._main_directory, self._mp3_directory))

    def _create_wav_dirs(self) -> None:
        for language in self._available_languages_mp3:
            os.makedirs(os.path.join(self._main_directory, self._wav_directory, language, 'clips'), exist_ok=True)

    def _load_available_clips(self) -> dict[str, list[str]]:
        try:
            return {
                language: os.listdir(os.path.join(self._main_directory, self._mp3_directory, language, 'clips'))
                for language in self._available_languages_mp3
            }
        except Exception as ex:
            print(ex)

    @staticmethod
    def remove_files(path: str) -> None:
        if os.path.exists(path):
            os.remove(path)

    def call_ffmpeg(self) -> None:
        try:
            self._create_wav_dirs()
            for language in self._available_languages_mp3:
                for filename in self._available_clips.get(language):
                    subprocess.call([
                        'ffmpeg',
                        '-y',
                        '-i',
                        os.path.join(
                            self._main_directory,
                            self._mp3_directory,
                            language,
                            'clips',
                            filename
                        ),
                        os.path.join(
                            self._main_directory,
                            self._wav_directory,
                            language,
                            'clips',
                            filename.replace('.mp3', '.wav')
                        )
                    ])
                    print('--OK--')
        except Exception as ex:
            print(ex)



