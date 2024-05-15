import pandas as pd
from src.hyperparams.merger import HparamsMerger


class AudioMetaInfo(HparamsMerger):
    def __init__(self, max_client_id_amount: int, min_clip_duration_ms: int, set_size: int, lang: str):
        super().__init__(max_client_id_amount, min_clip_duration_ms, set_size, lang)
        self._clips_duration_path: str = self.paths_info.DURATION_INFO_FILES_PATHS.get(lang)
        self._validates_path: str = self.paths_info.VALIDATED_INFO_FILES_PATHS.get(lang)

        self.audio_info = pd.read_csv(
            self._validates_path,
            sep='\t',
            usecols=['client_id', 'path', 'sentence_id', 'gender', 'locale']
        )

        self.df_men = None
        self.df_women = None

        self.initialize()


    def _filter_sex(self):
        man_filter = self.audio_info['gender'] == 'male_masculine'
        woman_filter = self.audio_info['gender'] == 'female_feminine'
        self.df_men = self.audio_info[man_filter]
        self.df_women = self.audio_info[woman_filter]

    def _add_duration_info(self) -> None:
        clips_duration = pd.read_csv(self._clips_duration_path, sep='\t')
        self.df_men = self.df_men.merge(clips_duration, left_on='path', right_on='clip')
        self.df_women = self.df_women.merge(clips_duration, left_on='path', right_on='clip')

    def _filter_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        rows_over = df[df['duration[ms]'] >= self.audio_hyperparameters.min_clip_duration_ms]
        rows_under = df[df['duration[ms]'] < self.audio_hyperparameters.min_clip_duration_ms]
        if len(rows_over) >= self.set_hyperparameters.set_size // 2:
            df = rows_over
        else:
            df = pd.concat([rows_over, rows_under], ignore_index=True)[:self.set_hyperparameters.set_size]
        return df

    def initialize(self):
        self._filter_sex()
        self._add_duration_info()
        self.df_men = self._filter_rows(self.df_men)
        self.df_women = self._filter_rows(self.df_women)

    def get_df_men(self) -> pd.DataFrame:
        return self.df_men

    def get_df_women(self) -> pd.DataFrame:
        return self.df_women

