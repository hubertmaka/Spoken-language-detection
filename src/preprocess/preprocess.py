import tensorflow as tf
import tensorflow_io as tfio
import random
from src.preprocess.audio_metafiles import AudioMetaInfo
from src.preprocess.split import SplitSet
from src.hyperparams.paths_info import PathsInfo
from src.preprocess.process_audio import ProcessAudio
from src.hyperparams.set_hyperparameters import SetHyperparameters


class Preprocess:
    SAMPLE_RATE = 48_000
    MAX_CLIENT_ID_AMOUNT = 1000
    MIN_CLIP_DURATION_MS = 4000
    SET_SIZE = 300

    SET_HPARAMS = SetHyperparameters(SET_SIZE)

    @staticmethod
    def add_zeros(wav, sample_rate):
        time_probes = wav.shape[0]
        missing_probes_one_side = int((Preprocess.MIN_CLIP_DURATION_MS / 1000 * sample_rate - time_probes) // 2)
        padded_tensor = tf.pad(wav.numpy(), [[missing_probes_one_side, missing_probes_one_side]])
        return tf.convert_to_tensor(padded_tensor, dtype=tf.float32)

    @staticmethod
    def cut_audio(wav, sample_rate):
        time_probes = wav.shape[0]
        overlap = int((time_probes - (Preprocess.MIN_CLIP_DURATION_MS / 1000) * sample_rate) / 2)
        cut_clip = wav[overlap:(time_probes - overlap)]
        return tf.convert_to_tensor(cut_clip, dtype=tf.float32)

    @staticmethod
    def load_audio(filename, fin_sam_rate=16_000):
        file_content = tf.io.read_file(filename)
        audio = tfio.audio.decode_mp3(file_content)
        audio = tf.squeeze(audio, axis=-1)
        return tf.convert_to_tensor(audio, dtype=tf.float32)

    @staticmethod
    def align_probes(audio, sample_rate):
        expected_probes = int((Preprocess.MIN_CLIP_DURATION_MS / 1000) * Preprocess.SAMPLE_RATE)
        current_probes = audio.shape[0]
        if expected_probes > current_probes:
            audio = Preprocess.add_zeros(audio, sample_rate)
        elif expected_probes < current_probes:
            audio = Preprocess.cut_audio(audio, sample_rate)
        return tf.convert_to_tensor(audio, dtype=tf.float32)

    @staticmethod
    def load_and_align_probes(file_path):
        audio = Preprocess.load_audio(file_path)
        expected_probes = int((Preprocess.MIN_CLIP_DURATION_MS / 1000) * Preprocess.SAMPLE_RATE)
        current_probes = audio.shape[0]
        if expected_probes > current_probes:
            audio = Preprocess.add_zeros(audio, Preprocess.SAMPLE_RATE)
        elif expected_probes < current_probes:
            audio = Preprocess.cut_audio(audio, Preprocess.SAMPLE_RATE)
        return tf.convert_to_tensor(audio, dtype=tf.float32)

    @staticmethod
    def process_random_samples(dataset, num_samples_to_process):
        processed_samples = []

        shuffled_dataset = dataset.shuffle(buffer_size=len(dataset))
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
    def preprocess():
        train_dataset = tf.data.Dataset.from_tensor_slices(([], []))
        val_dataset = tf.data.Dataset.from_tensor_slices(([], []))
        test_dataset = tf.data.Dataset.from_tensor_slices(([], []))

        languages = PathsInfo.get_languages()
        language_to_index = {lang: idx for idx, lang in enumerate(languages)}

        def one_hot_encode_language(lang):
            lang_index = language_to_index[lang]
            one_hot = tf.one_hot(lang_index, len(languages))
            return one_hot

        for lang in PathsInfo.get_languages():
            print(f"PROCESSED LANGUAGES: {lang}")
            audio_meta_info = AudioMetaInfo(
                max_client_id_amount=Preprocess.MAX_CLIENT_ID_AMOUNT,
                min_clip_duration_ms=Preprocess.MIN_CLIP_DURATION_MS,
                set_size=Preprocess.SET_SIZE,
                lang=lang
            )

            print(f"---AUDIO META INFO DONE {lang}---")

            df_men = audio_meta_info.get_df_men()
            df_women = audio_meta_info.get_df_women()

            for num, gender_df in enumerate([df_men, df_women]):
                df_filenames = SplitSet(
                    df=gender_df,
                    max_client_id_amount=Preprocess.MAX_CLIENT_ID_AMOUNT,
                    min_clip_duration_ms=Preprocess.MIN_CLIP_DURATION_MS,
                    set_size=Preprocess.SET_SIZE,
                    lang=lang
                ).get_filenames()

                print(f"---SET SPLIT DONE {num} {lang}---")

                train_filenames = df_filenames.get('train').apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()
                val_filenames = df_filenames.get('val').apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()
                test_filenames = df_filenames.get('test').apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()

                train_labels = [one_hot_encode_language(lang)] * len(train_filenames)
                val_labels = [one_hot_encode_language(lang)] * len(val_filenames)
                test_labels = [one_hot_encode_language(lang)] * len(test_filenames)

                train_dataset_tmp = tf.data.Dataset.zip((
                    tf.data.Dataset.from_tensor_slices(train_filenames),
                    tf.data.Dataset.from_tensor_slices(train_labels)
                ))
                val_dataset_tmp = tf.data.Dataset.zip((
                    tf.data.Dataset.from_tensor_slices(val_filenames),
                    tf.data.Dataset.from_tensor_slices(val_labels)
                ))
                test_dataset_tmp = tf.data.Dataset.zip((
                    tf.data.Dataset.from_tensor_slices(test_filenames),
                    tf.data.Dataset.from_tensor_slices(test_labels)
                ))

                print(f"---DATASET DONE {num} {lang}---")

                filled_samples = Preprocess.process_random_samples(
                    train_dataset_tmp, Preprocess.SET_HPARAMS.train_size - len(train_dataset_tmp)
                )

                filled_audio_samples = tf.data.Dataset.from_tensor_slices([Preprocess.align_probes(tensor[0], Preprocess.SAMPLE_RATE) for tensor in filled_samples])
                filled_labels = tf.data.Dataset.from_tensor_slices([tensor[1] for tensor in filled_samples])

                filled_samples = tf.data.Dataset.zip((filled_audio_samples, filled_labels))

                train_dataset_tmp = train_dataset_tmp.concatenate(filled_samples)

                train_dataset = train_dataset.concatenate(train_dataset_tmp)
                val_dataset = val_dataset.concatenate(val_dataset_tmp)
                test_dataset = test_dataset.concatenate(test_dataset_tmp)

                del filled_samples
                del train_dataset_tmp

            del df_men
            del df_women

        train_dataset = train_dataset.map(lambda audio, label: (ProcessAudio(audio, Preprocess.SAMPLE_RATE).normalize_audio(), label))
        val_dataset = val_dataset.map(lambda audio, label: (ProcessAudio(audio, Preprocess.SAMPLE_RATE).normalize_audio(), label))
        test_dataset = test_dataset.map(lambda audio, label: (ProcessAudio(audio, Preprocess.SAMPLE_RATE).normalize_audio(), label))

        train_dataset = train_dataset.map(lambda audio, label: (ProcessAudio(audio, Preprocess.SAMPLE_RATE).create_spectrogram(), label))
        val_dataset = val_dataset.map(lambda audio, label: (ProcessAudio(audio, Preprocess.SAMPLE_RATE).create_spectrogram(), label))
        test_dataset = test_dataset.map(lambda audio, label: (ProcessAudio(audio, Preprocess.SAMPLE_RATE).create_spectrogram(), label))

        print("CACHING")
        # train_dataset = train_dataset.cache()
        # val_dataset = val_dataset.cache()
        # test_dataset = test_dataset.cache()
        print("SHUFFLE")
        train_dataset = train_dataset.shuffle(buffer_size=1000)
        val_dataset = val_dataset.shuffle(buffer_size=1000)
        test_dataset = test_dataset.shuffle(buffer_size=1000)

        print("BATCHING")
        train_dataset = train_dataset.batch(16)
        val_dataset = val_dataset.batch(16)
        test_dataset = test_dataset.batch(16)
        print(train_dataset.as_numpy_iterator().next())
        print(val_dataset.as_numpy_iterator().next())
        print(test_dataset.as_numpy_iterator().next())

        return train_dataset, val_dataset, test_dataset

Preprocess.preprocess()
