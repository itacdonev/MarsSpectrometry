"""
FEATURE ENGINEERING
"""


import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from src import config, features, preprocess


class CreateFeatures:
    """
    Define features
    """

    def __init__(self,
                 metadata,
                 files_dict:dict=None,
                 file_suffix:str=None,
                 detrend_method:str='min',
                 remove_mz_cnt:bool=False,
                 remove_mz_thrs=None,
                 smooth:bool=False,
                 smoothing_type:str='gauss',
                 gauss_sigma:int=5,
                 ma_step:int=None):
        """
        Initialize a class
        metadata: metadata file of all samples
        files_dict: dictionary with (idx,file path) of
                    time series csv files
        """
        self.metadata = metadata
        self.files_dict = files_dict
        self.file_suffix = file_suffix
        self.detrend_method = detrend_method
        self.remove_mz_cnt = remove_mz_cnt
        self.remove_mz_thrs = remove_mz_thrs
        self.smooth = smooth
        self.smoothing_type = smoothing_type
        self.gauss_sigma = gauss_sigma
        self.ma_step = ma_step

    def __repr__(self):
        return "Creating features ... "


    # Combine more than one feature data set
    def combine_features(self,
                         features_list:list,
                         feature_name:str):
        """
        Combine more than one computed data frame
        into a one training sample.

        features_list (list): Should contain full names
                              of files to combine saved in
                              DATA_DIR_OUT.
        """
        if len(features_list) > 0:

            df_fts = pd.DataFrame()

            for fts in features_list:
                file_path = os.path.join(config.DATA_DIR_OUT, 
                                         fts + '_' + self.file_suffix + '.csv')
                if os.path.exists(file_path):
                    temp_df = pd.read_csv(file_path)
                    df_fts = pd.concat([df_fts, temp_df], axis=1)
                    assert temp_df.shape[0] == df_fts.shape[0]

        df_fts = df_fts.replace(np.nan, 0)

        # Save feature data frame
        df_fts.to_csv(os.path.join(config.DATA_DIR_OUT,
                                   feature_name + '_' + self.file_suffix + '.csv'),
                        index=False)

        return df_fts


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
        #print(colored(f'fts_corr_mz4 => {df_corr.shape}', 'blue'))
        assert df_corr.shape[0] == len(self.files_dict)

        return df_corr


    # TempBin+MZ = Max relative abundance
    def fts_mra_tempmz(self):
        """
        Bin temperature into 100 degree bins for each
        m/z values. Compute max relative abundance for
        each bin.
        """
        # Initialize a table to store computed values
        dt = pd.DataFrame(dtype='float64')
        
        #ion_temp_dict = {}
        
        # Loop over all sample_id and compute. Add computation to dt.
        print(f'Number of samples: {len(self.files_dict)}')
        for i in self.files_dict:
            sample_name = self.metadata.iloc[i]['sample_id']
            df_sample = preprocess.get_sample(self.metadata, i)
            #TODO all these don't have to be declaed for the class
            # but only within this function - FIX IT
            ht_pivot = features.bin_temp_abund(df_sample,
                                               sample_name,
                                               self.detrend_method,
                                               self.remove_mz_cnt,
                                               self.remove_mz_thrs,
                                               self.smooth,
                                               self.smoothing_type,
                                               self.gauss_sigma,
                                               self.ma_step)
            #ion_temp_dict[sample_name] = ht_pivot
            dt = pd.concat([dt, ht_pivot])
        
        dt = dt.set_index('sample_id')
        
        # Rename columns
        t_cols = dt.columns
        remove_chars = "(,]"
        for char in remove_chars:
            t_cols = [i.replace(char,'') for i in t_cols]
        t_cols = [i.replace(' ','_') for i in t_cols]
        dt.columns = t_cols
        
        # Replace NAN values
        dt = dt.replace(np.nan, 0)
        
        # Save features file to csv
        if self.file_suffix:
            dt.to_csv(os.path.join(config.DATA_DIR_OUT,
                                        'fts_mra_tempmz_' + self.file_suffix + '.csv'),
                        index=False)
        else:
            dt.to_csv(os.path.join(config.DATA_DIR_OUT,
                                        'fts_mra_tempmz_.csv'),
                        index=False)
            
        return dt
    
    
    def fts_cntpk_mratt(self):
        """
        Combines all computed ion peaks stats from each sample
        into a features data frame.
        """
        # Initialize a data frame to store all the sample calculations
        df = pd.DataFrame()

        for sample_idx in tqdm(self.files_dict):
            ion_peaks_df = features.compute_ion_peaks(metadata=self.metadata,
                                                      sample_idx=sample_idx,
                                                      detrend_method=self.detrend_method,
                                                      gauss_sigma=self.gauss_sigma)
            df = pd.concat([df,ion_peaks_df], axis = 0)

        # Join multi column index into one separated by -
        df.columns = df.columns.map(lambda x: '_'.join([str(i) for i in x]))

        # Save features file to csv
        if self.file_suffix:
            df.to_csv(os.path.join(config.DATA_DIR_OUT,
                                        'fts_cntpk_mratt_' + self.file_suffix + '.csv'),
                        index=False)
        else:
            df.to_csv(os.path.join(config.DATA_DIR_OUT,
                                        'fts_cntpk_mratt_.csv'),
                        index=False)

        return df
    