"""
FEATURE ENGINEERING
"""


import os
from tqdm import tqdm
import pandas as pd
from termcolor import colored
from src import config, features, preprocess


class CreateFeatures:
    """
    Define features
    """

    def __init__(self, metadata, files_dict:dict, file_suffix:str=None):
        """
        Initialize a class
        metadata: metadata file of all samples
        files_dict: dictionary with (idx,file path) of
                    time series csv files
        """
        self.metadata = metadata
        self.files_dict = files_dict
        self.file_suffix = file_suffix

    def __repr__(self):
        return "Creating features ... "

    # Correlation mz4 with other mz values
    def fts_corr_mz4(self):
        """
        Compute linear relationship between
        correlation of mz=4 and other mz values.
        """
        mz4_corr_dict = {}
        for file_idx in tqdm(self.files_dict):
            df_sample = preprocess.get_sample(self.metadata, file_idx)
            df_sample = preprocess.preprocess_samples(df_sample,
                                                    detrend_method='min',
                                                    remove_He=False)

            mz4_corr = features.lr_corr_mz4(df_sample)
            mz4_corr_dict[file_idx] = mz4_corr

        df_corr = pd.DataFrame.from_dict(mz4_corr_dict, orient='index')
        df_corr.columns = ['lr_corr_mz4']

        # Save features file to csv
        if self.file_suffix:
            df_corr.to_csv(os.path.join(config.DATA_DIR_OUT,
                                        'fts_corr_mz4_' + self.file_suffix + '.csv'),
                        index=False)
        else:
            df_corr.to_csv(os.path.join(config.DATA_DIR_OUT,
                                        'fts_corr_mz4.csv'),
                        index=False)
        print(colored(f'fts_corr_mz4 => {df_corr.shape}', 'blue'))
        assert df_corr.shape[0] == len(self.files_dict)

        return df_corr
    