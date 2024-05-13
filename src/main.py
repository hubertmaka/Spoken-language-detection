from src.utils import ConvertMp3ToWav


def main() -> None:
    ConvertMp3ToWav(
        'languages_audio',
        'languages_audio_mp3',
        'languages_audio_wav').call_ffmpeg()


if __name__ == "__main__":
    main()
