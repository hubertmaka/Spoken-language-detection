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
    SET_SIZE = 800

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
    def align_probes(audio, sample_rate):
        expected_probes = int((Preprocess.MIN_CLIP_DURATION_MS / 1000) * Preprocess.SAMPLE_RATE)
        print("EXPECTED", expected_probes)
        current_probes = audio.shape[0]
        print("CURRENT", current_probes)
        if expected_probes > current_probes:
            print("ADD ZEROS")
            return Preprocess.add_zeros(audio, sample_rate)
        elif expected_probes < current_probes:
            print("CUT AUDIO")
            return Preprocess.cut_audio(audio, sample_rate)
        return tf.convert_to_tensor(audio, dtype=tf.float32)

    @staticmethod
    def load_and_align_probes(file_path):
        audio = Preprocess.load_audio(file_path)
        expected_probes = int((Preprocess.MIN_CLIP_DURATION_MS / 1000) * Preprocess.SAMPLE_RATE)
        current_probes = audio.shape[0]
        if expected_probes > current_probes:
            return Preprocess.add_zeros(audio, Preprocess.SAMPLE_RATE)
        elif expected_probes < current_probes:
            return Preprocess.cut_audio(audio, Preprocess.SAMPLE_RATE)
        return tf.convert_to_tensor(audio, dtype=tf.float32)

    @staticmethod
    def process_random_samples(dataset, num_samples_to_process):
        processed_samples = []

        shuffled_dataset = dataset.shuffle(buffer_size=len(dataset))
        samples = shuffled_dataset.take(num_samples_to_process)
        for sample in samples:
            sample = ProcessAudio(sample, Preprocess.SAMPLE_RATE).add_noise_rand()
            sample = ProcessAudio(sample, Preprocess.SAMPLE_RATE).change_amplitude_rand()
            sample = random.choice([

                ProcessAudio(sample, Preprocess.SAMPLE_RATE).time_masking_rand(),
            ])
            processed_samples.append(sample)
        print(processed_samples)
        return processed_samples

    @staticmethod
    def preprocess():
        train_dataset = tf.data.Dataset.from_tensor_slices([])
        val_dataset = tf.data.Dataset.from_tensor_slices([])
        test_dataset = tf.data.Dataset.from_tensor_slices([])
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

                train_dataset_tmp = tf.data.Dataset.from_tensor_slices(
                    df_filenames.get('train').apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()
                )
                val_dataset_tmp = tf.data.Dataset.from_tensor_slices(
                    df_filenames.get('val').apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()
                )
                test_dataset_tmp = tf.data.Dataset.from_tensor_slices(
                    df_filenames.get('test').apply(lambda filename: Preprocess.load_and_align_probes(filename)).to_list()
                )

                print(f"---DATASET DONE DONE {num} {lang}---")

                filled_samples = Preprocess.process_random_samples(
                    train_dataset_tmp, Preprocess.SET_HPARAMS.train_size - len(train_dataset_tmp)
                )
                print([sample.shape for sample in filled_samples])
                filled_samples = [Preprocess.align_probes(tensor, Preprocess.SAMPLE_RATE) for tensor in filled_samples]
                print([sample.shape for sample in filled_samples])
                filled_samples = tf.data.Dataset.from_tensor_slices(filled_samples)

                train_dataset_tmp = train_dataset_tmp.concatenate(filled_samples)

                print(f"---DATASET TRAIN FILL DONE {num} {lang}---")

                train_dataset = train_dataset.concatenate(train_dataset_tmp)
                val_dataset = val_dataset.concatenate(val_dataset_tmp)
                test_dataset = test_dataset.concatenate(test_dataset_tmp)

                del train_dataset_tmp

            del df_men
            del df_women

        train_dataset = train_dataset.map(lambda audio: ProcessAudio(audio, Preprocess.SAMPLE_RATE).normalize_audio())
        val_dataset = val_dataset.map(lambda audio: ProcessAudio(audio, Preprocess.SAMPLE_RATE).normalize_audio())
        test_dataset = test_dataset.map(lambda audio: ProcessAudio(audio, Preprocess.SAMPLE_RATE).normalize_audio())

        train_dataset = train_dataset.map(lambda audio: ProcessAudio(audio, Preprocess.SAMPLE_RATE).create_spectrogram())
        val_dataset = val_dataset.map(lambda audio: ProcessAudio(audio, Preprocess.SAMPLE_RATE).create_spectrogram())
        test_dataset = test_dataset.map(lambda audio: ProcessAudio(audio, Preprocess.SAMPLE_RATE).create_spectrogram())

        train_dataset = tf.data.Dataset.zip((
            train_dataset, tf.data.Dataset.from_tensor_slices(tf.ones(len(train_dataset)))
        ))
        val_dataset = tf.data.Dataset.zip((val_dataset, tf.data.Dataset.from_tensor_slices(tf.ones(len(val_dataset)))
        ))
        test_dataset = tf.data.Dataset.zip((test_dataset, tf.data.Dataset.from_tensor_slices(
            tf.ones(len(test_dataset)))))

        print(train_dataset.as_numpy_iterator().next())
        print(val_dataset.as_numpy_iterator().next())
        print(test_dataset.as_numpy_iterator().next())



Preprocess.preprocess()