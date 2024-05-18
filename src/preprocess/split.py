import os
import pandas as pd
from src.hyperparams.merger import HparamsMerger


class SplitSet(HparamsMerger):
    def __init__(self, df: pd.DataFrame, max_client_id_amount: int, min_clip_duration_ms: int, set_size: int, lang: str) -> None:
        super().__init__(max_client_id_amount, min_clip_duration_ms, set_size, lang)
        self.df = df
        self.counted_id = self.df['client_id'].value_counts(ascending=True)

        self._df_train = None
        self._df_val = None
        self._df_test = None

        self._split_set()

    def _split_set(self) -> None:
        df_train = pd.DataFrame()
        df_val = pd.DataFrame()
        df_test = pd.DataFrame()

        for i in range(len(self.counted_id)):
            rows_form_origin_df = self.df[self.df['client_id'] == self.counted_id.index[i]][:self.audio_hyperparameters.max_client_id_amount]
            if rows_form_origin_df['client_id'].iloc[0]:
                if len(rows_form_origin_df) <= (self.set_hyperparameters.test_size - len(df_test)):
                    df_test = pd.concat([df_test, rows_form_origin_df], ignore_index=True)
                    continue
                if len(df_test) < self.set_hyperparameters.test_size:
                    df_test = pd.concat([df_test, rows_form_origin_df[:(self.set_hyperparameters.test_size - len(df_test))]], ignore_index=True)
                    continue
                if len(rows_form_origin_df) <= (self.set_hyperparameters.val_size- len(df_val)):
                    df_val = pd.concat([df_val, rows_form_origin_df], ignore_index=True)
                    continue
                if len(df_val) < self.set_hyperparameters.val_size:
                    df_val = pd.concat([df_val, rows_form_origin_df[:(self.set_hyperparameters.val_size - len(df_val))]], ignore_index=True)
                    continue
                if len(rows_form_origin_df) <= (self.set_hyperparameters.train_size - len(df_train)):
                    df_train = pd.concat([df_train, rows_form_origin_df], ignore_index=True)
                    continue
                if len(df_train) < self.set_hyperparameters.train_size:
                    df_train = pd.concat([df_train, rows_form_origin_df[:(self.set_hyperparameters.train_size - len(df_train))]], ignore_index=True)
                    continue

        self._df_train = df_train
        self._df_val = df_val
        self._df_test = df_test

    def get_filenames(self) -> dict[str, pd.DataFrame]:
        return {
            'train': self._df_train['path'].apply(lambda fn: os.path.join(self.paths_info.LANG_DIRS.get(self.lang), 'clips', fn)),
            'val': self._df_val['path'].apply(lambda fn: os.path.join(self.paths_info.LANG_DIRS.get(self.lang),'clips', fn)),
            'test': self._df_test['path'].apply(lambda fn: os.path.join(self.paths_info.LANG_DIRS.get(self.lang),'clips', fn))
        }

    def get_sets(self) -> dict[str, pd.DataFrame]:
        return {
            'train': self._df_train,
            'val': self._df_val,
            'test': self._df_test
        }


