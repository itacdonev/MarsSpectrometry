"""
FEATURE ENGINEERING
"""


import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src import config, features, preprocess


class CreateFeatures:
    """
    Define features
    """

    def __init__(self,
                 metadata,
                 files_dict,
                 file_suffix,
                 fts_name:str,
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
        self.fts_name = fts_name
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
                         features_list:list):
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
                                   self.fts_name + '_' + self.file_suffix + '.csv'),
                        index=False)

        return df_fts


    # Slope of correlation mz4 with other mz values
    def fts_lr_corr_mz4(self):
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
                                        self.fts_name + '_' + self.file_suffix + '.csv'),
                        index=False)
        else:
            df_corr.to_csv(os.path.join(config.DATA_DIR_OUT,
                                        self.fts_name + '_' + '.csv'),
                        index=False)
        #print(colored(f'fts_corr_mz4 => {df_corr.shape}', 'blue'))
        assert df_corr.shape[0] == len(self.files_dict)

        return df_corr

    # Correlation of mz4 and other mz values
    def fts_corr_mz4(self):
        """Correlation between mz4 and other mz ratios.
        """
        df = pd.DataFrame()

        for idx in tqdm(self.files_dict):
            df_sample = preprocess.get_sample(self.metadata, idx)
            df_sample = preprocess.preprocess_samples(df_sample,
                                                    detrend_method='min',
                                                    remove_He=False)
            sample_name = self.metadata.iloc[idx]['sample_id']
            df_corr = features.corr_mz4(df_sample, sample_name)
            df = pd.concat([df, df_corr], axis=0)
        
        df = df.replace(np.nan, 0)
        
        # Save file
        df.to_csv(os.path.join(config.DATA_DIR_OUT,
                               self.fts_name + '_' + self.file_suffix + '.csv'),
                        index=False)
        
        return df
        
    
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
                                        self.fts_name + '_' + self.file_suffix + '.csv'),
                        index=False)
        else:
            dt.to_csv(os.path.join(config.DATA_DIR_OUT,
                                        self.fts_name + '_' + '.csv'),
                        index=False)
            
        return dt
    
    # Count no peaks per mz, mra, time and temp at 1st peak  
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

        # Replace NaN values
        df = df.replace(np.nan, 0)

        # Save features file to csv
        if self.file_suffix:
            df.to_csv(os.path.join(config.DATA_DIR_OUT,
                                        self.fts_name + '_' + self.file_suffix + '.csv'),
                        index=False)
        else:
            df.to_csv(os.path.join(config.DATA_DIR_OUT,
                                        self.fts_name + '_' + '.csv'),
                        index=False)

        return df

    # Slope of time ~ temp
    def fts_slope_tt(self):
        """
        Compute slope of time ~ temp for each sample using
        linear regression. Is there any difference between
        commercial and sam_testbed samples?
        """

        coefs_lr = {}

        for i in tqdm(self.files_dict):
            ht = preprocess.get_sample(self.metadata, i)
            ht = preprocess.preprocess_samples(ht, detrend_method=self.detrend_method)
            sample_name = self.metadata.iloc[i]['sample_id']

            lr = LinearRegression()

            X = np.array(ht['time']).reshape(-1, 1)
            y = ht['temp'].values
            lr.fit(X, y)

            coefs_lr[sample_name] = lr.coef_[0]

        df_slope = pd.DataFrame.from_dict(coefs_lr, orient='index')
        df_slope.columns = ['slope_tt']
        
        # Save feature data frame
        df_slope.to_csv(os.path.join(config.DATA_DIR_OUT,
                                   self.fts_name + '_' + self.file_suffix + '.csv'),
                        index=False)
        
        return df_slope

    # Spectra information per sample
    def fts_spectra(self): #features_ms
        """
        Features from mass spectra.
        """
        df = pd.DataFrame()
        
        for file_idx in tqdm(self.files_dict): #idx
            #print(file_idx)
            temp = {}
            
            # Select a sample and get sample name
            df_sample = preprocess.get_sample(self.metadata, file_idx)
            sample_name = self.metadata.iloc[file_idx]['sample_id']
            temp['sample_id'] = sample_name
            
            # Preprocess the sample
            df_sample = preprocess.preprocess_samples(df_sample, 
                                                    detrend_method=self.detrend_method)
            
            # Molecular ion and its abundance
            M, M_mra = features.get_molecular_ion(df_sample)
            temp['M'] = M
            temp['M_mra'] = M_mra 
            temp = pd.DataFrame(data=temp, index=[0])
            
            # Isotopic abundance [M+1]
            Mp1_mra, Mp1_isotope = features.get_Mp1_peak(df_sample)
            temp['Mp1_mra'] = Mp1_mra
            temp['Mp1_isotope'] = np.round(Mp1_isotope,2)
            
            # Isotopic abundance [M+2]
            Mp2_mra, Mp2_isotope = features.get_Mp2_peak(df_sample)
            temp['Mp2_mra'] = Mp2_mra
            temp['Mp2_isotope'] = np.round(Mp2_isotope,2)
            
            
            # 1st heavy isotopes
            if np.round(Mp1_isotope,3) in pd.Interval(0.014,0.016, closed='both'):
                temp['hi1'] = 1. #'H'
            elif np.round(Mp1_isotope,1) in pd.Interval(1.0,1.2, closed='both'):
                temp['hi1'] =  12. #'C'
            elif np.round(Mp1_isotope,2) in pd.Interval(0.36,0.38, closed='both'):
                temp['hi1'] =  14. #'N'
            elif np.round(Mp1_isotope,2) in pd.Interval(0.03,0.05, closed='both'):
                temp['hi1'] =  16. #'O'
            elif np.round(Mp1_isotope,1) in pd.Interval(5.0,5.2, closed='both'):
                temp['hi1'] =  28. #'Si'
            elif np.round(Mp1_isotope,1) in pd.Interval(0.7,0.9, closed='both'):
                temp['hi1'] =  32. #'S'
            else: temp['hi1'] = ''

            # 2nd heavy isotopes
            if np.round(Mp2_isotope,1) in pd.Interval(0.1, 0.3, closed='both'):
                temp['hi2'] = 16. #'O'
            elif np.round(Mp2_isotope,1) in pd.Interval(3.3, 3.5, closed='both'):
                temp['hi2'] =  28. #'Si'
            elif np.round(Mp2_isotope,1) in pd.Interval(4.3, 4.5, closed='both'):
                temp['hi2'] =  32. #'S'
            elif np.round(Mp2_isotope,1) in pd.Interval(32.4,32.6, closed='both'):
                temp['hi2'] =  35.5 #'Cl'
            elif np.round(Mp2_isotope,1) in pd.Interval(97.9,98.1, closed='both'):
                temp['hi2'] =  79.9 #'Br'
            else: temp['hi2'] = ''

            #TODO Adjusted M+1 and M+2

            # Number of carbons
            #TODO Needs correction if there are present 1st and 2nd isotopes
            temp['C_cnt'] = features.get_no_carbons(df_sample)            

            # Presence of nitrogen
            # an even molecular ion indicates the sample lacks nitrogen
            #N_cnt = nitrogen_rule(df_sample)
            #temp['N_cnt'] = N_cnt

            # Difference between molecular ion and its first fragment peak
            mi_frg_diff = features.get_moli_frag_peak_diff(df_sample)
            temp['mi_frg_diff'] = mi_frg_diff

            df = pd.concat([df, temp], axis=0)
            del df['sample_id']

            # Save feature data frame
            df.to_csv(os.path.join(config.DATA_DIR_OUT,
                                    self.fts_name + '_' + self.file_suffix + '.csv'),
                            index=False)

        return df

    # Abundance area of sample
    def fts_area_sample(self):
        """
        Compute the area under the smoothed abundance
        curve for a sample.
        """
        areas_dict = {}
        
        for idx in tqdm(self.files_dict):
            sample_name = self.metadata.iloc[idx]['sample_id']
            area_sample = features.sample_abund_area(self.metadata, 
                                                    idx, 
                                                    self.detrend_method)
            areas_dict[idx] = area_sample
            
        # Convert into a dataframe and save
        df_area = pd.DataFrame.from_dict(areas_dict, orient='index')
        df_area.columns = ['area']
        
        # Save feature data frame
        df_area.to_csv(os.path.join(config.DATA_DIR_OUT,
                                    self.fts_name + '_' + self.file_suffix + '.csv'),
                         index=False)
        
        return df_area

    #TODO Contribution of tempbin area to the whole area
    def fts_area_contrib_temp(self):
        """
        - Divide the temp into bins
        - Compute area for each bin.
        - Compute percentage area of the temp bin in terms of 
          total area - contribution of the temp bin to total m/z 
          value in area.
        """
        # features.area_contrib_temp()
        pass

    # Range abundance/temperature in each temp bin
    def fts_range_abun_to_temp(self):
        """
        Compute ratio of abundance to temperature for each 
        observation in the sample. Then we divide the temperature
        into bins and compute the range of the computed ratio
        for each temp bin.
        """
        df = pd.DataFrame(()
                          )
        for idx in tqdm(self.files_dict):
            df_range_att_sample = features.range_abun_to_temp(self.metadata,
                                                              idx,
                                                              self.detrend_method)

            df = pd.concat([df, df_range_att_sample], axis=0)
            
        # Save the data set
        df.to_csv(os.path.join(config.DATA_DIR_OUT,
                                    self.fts_name + '_' + self.file_suffix + '.csv'),
                         index=False)

        df = df.replace(np.nan, 0)

        return df

        
        
        
    # Statistics of each mz in the sample: mean, median, std, kurt, skew
    def fts_mzstats(self,):
        """
        Statistics of each mz in the sample: mean, median, std, kurt, skew
        """
        df = pd.DataFrame()
    
        for file_idx in tqdm(self.files_dict): #idx
            #print(file_idx)
            temp = {}
            
            # Select a sample and get sample name
            df_sample = preprocess.get_sample(self.metadata, file_idx)
            sample_name = self.metadata.iloc[file_idx]['sample_id']
            temp['sample_id'] = sample_name
            
            # Preprocess the sample
            df_sample = preprocess.preprocess_samples(df_sample,
                                                    detrend_method=self.detrend_method,
                                                    smooth=self.smooth)
            
            # Statistics
            sample_stats = features.get_mz_stats(df_sample, sample_name, self.smooth)
            
            # Fix columns names
            sample_stats['m/z'] = ['mz_'+str(i) for i in sample_stats['m/z']]
            sample_stats['m/z'] = [i.removesuffix(".0") for i in sample_stats['m/z']]

            # Prepare the df
            stats_pivot = sample_stats.pivot(index='sample_id', columns='m/z')
            df = pd.concat([df, stats_pivot], axis=0)
        
        df = df.fillna(0)
        df.columns = df.columns.map(lambda x: '_'.join([str(i) for i in x]))
        
        # Save feature data frame
        df.to_csv(os.path.join(config.DATA_DIR_OUT,
                                    self.fts_name + '_' + self.file_suffix + '.csv'),
                         index=False)

        return df


    # TopN mz ratios
    def fts_topmz(self,
                  N_values:int=3,
                  normalize:bool=True,
                  lb:float=0.0, ub:float=0.99):
        """
        Compute top N ions by their max relativeabundance.

        Parameters
        ----------
        N: int (default=3)
                Number of top ions to store

            normalize: str (default=True)
                Should the values be normalized
                
            up: float (default=0.99)
                Upper bound of feature range.
            
            lb: float (default=0.0)
                Lower bound of feature range.
                
        Returns
        -------
            dictionary of sample and a list of top N ions
        """
        
        top3_ions = {} # sample, list of top 3 ions - {'S0000',[18.0, 9.0, 25.0]}
        
        # Compute for each sample in metadata
        for i in tqdm(self.files_dict):
            # Get and preprocess data sample
            df_sample = preprocess.get_sample(self.metadata, i)
            df_sample = preprocess.preprocess_samples(df_sample,
                                                      detrend_method=self.detrend_method,
                                                      smooth=self.smooth)
            sample_name = self.metadata.iloc[i]['sample_id']

            # Compute top 3 ions by relative abundance
            # Take max of each ion group sort and slice top N
            top3 = list((df_sample.groupby('m/z')['abun_scaled']\
                            .agg('max')\
                            .sort_values(ascending=False))\
                                .head(N_values).index)

            top3_ions[sample_name] = top3

        # Convert to data frame
        temp = pd.DataFrame.from_dict(top3_ions, orient='index')
        # Rename columns
        temp.columns = ['top_%s' % (i+1) for i in range(N_values)]

        # Normalize values
        if normalize:
            # Compute min,max for all columns
            minv = (temp.min()).min()
            maxv = (temp.max()).max()
            for col in temp:
                col_std = (temp[col] - minv) / (maxv - minv)
                temp[col] = col_std * (ub - lb) + lb

        # Fix the index
        #temp.index = temp.index.set_names('sample_id')
        #temp = temp.reset_index()

        # Save feature data frame
        temp.to_csv(os.path.join(config.DATA_DIR_OUT,
                                    self.fts_name + '_' + self.file_suffix + '.csv'),
                         index=False)

        return temp



    # Temp IQ4 stats per mz ratio
    def fts_tempIQ_mzstats(self):
        """
        Divide temperature in each sample into 4 IQ bins and compute stats for all such as:
            - temp range
            - area
            - area - percent to total
            - number of peaks
            - max rel abund
            - variance
            - std
            - time range
        """
        pass
    
    
    # Peak widths
    def fts_peak_widths(self, no_peaks_calc):
        
        # Create data frame to store calculation on sample level
        width_names = ['perc'+str(i) for i in ['10', '50', '90']]
        mz_names = ['mz' + str(i) for i in range(0,100,1)]
        peak_names = ['peak'+str(i) for i in range(1,no_peaks_calc+1,1)]
        column_names = []
        for w in width_names:
            for mz in mz_names:
                for p in peak_names:
                    column_names.append(w+'_'+mz+'_'+p)
        df_all_samples_width = pd.DataFrame(columns=column_names)
        
        # Select one sample
        for idx in tqdm(self.files_dict): #idx
            df_sample = preprocess.get_sample(self.metadata,idx)
            df_sample = preprocess.preprocess_samples(df_sample,
                                                    detrend_method='min')
            # Get sample name
            sample_name = self.metadata.iloc[idx]['sample_id']
            #print(f'Sample: {sample_name}')

            sample_widths = pd.DataFrame()
            # Compute for each mz ratio
            for mz in df_sample['m/z'].unique().tolist():
                #print(f'MZ: {mz}')
                df_mz_width = features.peak_width_mz(self.metadata,
                                                    df_sample,
                                                    idx,
                                                    mz,
                                                    no_peaks_calc,
                                                    make_plot=False)
                # Add one MZ calculation
                sample_widths = pd.concat([sample_widths, df_mz_width],
                                          axis=1)
            #Add one sample calculation
            if sample_widths.empty:
                dfempty = pd.DataFrame([[np.nan] * df_all_samples_width.shape[1]],
                                       columns=df_all_samples_width.columns)
                dfempty['sample_id'] = sample_name
                dfempty = dfempty.set_index('sample_id')
                df_all_samples_width = pd.concat([df_all_samples_width,
                                                  dfempty],
                                                 axis=0)
            else:
                df_all_samples_width = pd.concat([df_all_samples_width,
                                                sample_widths],
                                                axis=0)

        df_all_samples_width = df_all_samples_width.replace(np.nan, 0)

        # Save feature data frame
        df_all_samples_width.to_csv(os.path.join(config.DATA_DIR_OUT,
                                    self.fts_name + '_' + self.file_suffix + '.csv'),
                         index=False)
        
        return df_all_samples_width
