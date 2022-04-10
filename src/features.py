"""Feature engineering"""

import gc
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from src import config, preprocess, utils


# BENCHMARK FEATURES
# Bin the temp and compute processes max abundance for each bin
def bin_temp_abund(df_sample, sample_name:str, detrend_method:str,
                   remove_mz_cnt:bool=False, remove_mz_thrs=None,
                   smooth:bool=False,
                    smoothing_type:str='gauss',
                    gauss_sigma:int=5,
                    ma_step:int=None):
    """
    Create equal bins for temperature
    Compute max for relative abundance for each bin and ion type
    Return a dataframe where rows are samples and columns are temp-ion bins
    Computed on sample level
    """
    # Create a series of temperature bins
    temprange = pd.interval_range(start=-100, end=1500, freq=100)
    df_sample['temp_bin'] = pd.cut(df_sample['temp'], bins=temprange)
    
    # Preprocess the sample data
    df_sample = preprocess.preprocess_samples(df_sample, 
                                              detrend_method=detrend_method,
                                              remove_mz_cnt=remove_mz_cnt,
                                              remove_mz_thrs=remove_mz_thrs,
                                              smooth=smooth,
                                              smoothing_type=smoothing_type,
                                                gauss_sigma=gauss_sigma,
                                                ma_step=ma_step)
    
    # Compute max relative abundance
    if smooth:
        ht = df_sample.groupby(['m/z', 'temp_bin'])['abun_scaled_smooth'].agg('max').reset_index()
    else:
        ht = df_sample.groupby(['m/z', 'temp_bin'])['abun_scaled'].agg('max').reset_index()
    
    ht = ht.replace(np.nan, 0)
    ht['sample_id'] = sample_name
    ht.columns = ['Ion_type', 'temp_bin', 'max_rel_abund', 'sample_id']
    
    ht_pivot = ht.pivot(index='sample_id', 
                columns=['Ion_type', 'temp_bin'],
                values='max_rel_abund')
    ht_pivot.columns = ht_pivot.columns.map(lambda x: '_'.join([str(i) for i in x]))
    ht_pivot = ht_pivot.add_prefix('Ion_')
    ht_pivot = ht_pivot.reset_index().rename_axis(None, axis=1)

    #ht_pivot.drop('sample_id', axis = 1, inplace=True)
    
    return ht_pivot


def features_iontemp_abun(df_meta, 
                          sample_list, 
                          detrend_method:str,
                          remove_mz_cnt:bool=False, 
                          remove_mz_thrs=None,
                          smooth:bool=False,
                          smoothing_type:str='gauss',
                          gauss_sigma:int=5,
                          ma_step:int=None):
    """
    Finish
    """
    # Initialize a table to store computed values
    dt = pd.DataFrame(dtype='float64')
    
    #ion_temp_dict = {}
    
    # Loop over all sample_id and compute. Add computation to dt.
    print(f'Number of samples: {len(sample_list)}')
    for i in sample_list:
        sample_name = df_meta.iloc[i]['sample_id']
        #print(sample_name)
        df_sample = preprocess.get_sample(df_meta, i)
        
        ht_pivot = bin_temp_abund(df_sample, sample_name, detrend_method,
                                  remove_mz_cnt=remove_mz_cnt,
                                  remove_mz_thrs=remove_mz_thrs,
                                  smooth=smooth,
                                    smoothing_type=smoothing_type,
                                    gauss_sigma=gauss_sigma,
                                    ma_step=ma_step)
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

    return dt
     

def filter_areas_tempion(df, area_thrs):
    """
    Filter all the ions where the areas in all
    of the temperature bins is less than area_thrs.
    """
    dft = df.copy()
    #print(f'Original shape: {dft.shape}')
    for col in dft:
        if all(dft[col] < area_thrs):
            #dft[col] = 0
            del dft[col]
    #print(f'Final shape: {dft.shape}')
    
    return dft


def bin_temp_area(df_sample, sample_name, detrend_method:str, 
                  to_pivot:bool=True, filter_ions:bool=False, 
                  area_thrs:float=1.0):
    """
    Compute area for the temp-ion bin.
    """
    # Create a series of temperature bins
    temprange = pd.interval_range(start=-100, end=1500, freq=100)
    df_sample['temp_bin'] = pd.cut(df_sample['temp'], bins=temprange)
    
    # Preprocess the sample data
    df_sample = preprocess.preprocess_samples(df_sample, 
                                              detrend_method=detrend_method)
    
    ion_list = df_sample['m/z'].unique().tolist()
    bin_list = df_sample['temp_bin'].unique().tolist()
    
    # Loop over all m/z ions:
    ion_areas_dict = {}
    for ion in ion_list:    
        tempdf = df_sample[df_sample['m/z'] == ion].copy()
        bin_areas_dict = {}
        for bin in bin_list:
            dtt = tempdf[tempdf.temp_bin == bin].copy()
            # Area of the abundance
            dtt = dtt.sort_values(by=['temp', 'abun_scaled'])
            x = dtt['temp'].values
            y = dtt['abun_scaled'].values
            area = np.trapz(y=y,x=x)
            bin_areas_dict[bin] = area
        ion_areas_dict[ion] = bin_areas_dict
    
    if to_pivot:
        # Prepare df so that each row is sample id
        df_pivot = pd.DataFrame.from_dict(ion_areas_dict)
        df_pivot.index = df_pivot.index.set_names('temp_bin')
        
        if filter_ions:
            df_pivot = filter_areas_tempion(df_pivot, area_thrs)
        
        df_pivot = df_pivot.reset_index()
        df_pivot = df_pivot.melt('temp_bin')
        df_pivot['sample_id'] = sample_name
        df_pivot['temp_bin'] = df_pivot['temp_bin'].astype('str')
        df_pivot = df_pivot.pivot(index='sample_id', 
                                columns=['variable', 'temp_bin'], 
                                values='value')
        df_pivot.columns = df_pivot.columns.map(lambda x: '_'.join([str(i) for i in x]))
        df_pivot = df_pivot.add_prefix('Ion_')
        df_pivot = df_pivot.reset_index().rename_axis(None, axis=1)
        return df_pivot
    else:
        return ion_areas_dict
    


def features_iontemp_area(df_meta, sample_list, detrend_method:str,
                          filter_ions:bool=False, area_thrs:float=1.0):
    # Initialize a table to store computed values
    dt = pd.DataFrame(dtype='float64')
    
    # Loop over all sample_id and compute. Add computation to dt.
    #print(f'Number of samples: {len(sample_list)}')
    for i in sample_list:
        #print(f'Sample: {i}')
        sample_name = df_meta.iloc[i]['sample_id']
        #print(sample_name)
        df_sample = preprocess.get_sample(df_meta, i)
        
        ht_pivot = bin_temp_area(df_sample, sample_name, detrend_method,
                                 filter_ions, area_thrs)
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

    return dt.fillna(0)
    
    
    
# ===== Duration to max temperature per ion =====
def ion_duration_maxtemp(df_sample, ion_list, detrend_method):
    
    # Preprocess the sample data
    df_sample = preprocess.preprocess_samples(df_sample, detrend_method=detrend_method)

    # Initialize dictionary
    ion_time_dict = {}
    
    for i in ion_list:
        # Compute only for ions present in the sample
        if i in df_sample['m/z'].unique():
            # Select the ion type
            ht = df_sample[df_sample['m/z'] == i].copy()
            ion_time_dict[i] = ht[ht.abun_minsub_scaled == max(ht.abun_minsub_scaled)]['time'].iloc[0]
        else:
            ion_time_dict[i] = np.nan
    
    return ion_time_dict
    
def features_ion_duration_maxtemp(df_meta, file_paths, ion_list):
    
    # Data frame to save results
    fts_df = pd.DataFrame(data = {'m/z': np.arange(0,100,1.)}, 
                          dtype='float')
    
    for i in tqdm(file_paths):
        # Get sample
        df_sample = preprocess.get_sample(df_meta, i)
        sample_name = df_meta.loc[i, 'sample_id']
        
        # Compute duration to max abundance
        duration_max_abund = ion_duration_maxtemp(df_sample, ion_list)
    
        # Append values to data frame
        fts_df[sample_name] = fts_df['m/z'].map(duration_max_abund)
        gc.collect()
        
    # Transpose the data frame
    df = fts_df.iloc[:,1:].T
    df.columns = fts_df['m/z'].astype('str')
    df = df.add_prefix('Ion_')
    del df['Ion_4.0']
    
    return df



        
# ===== SPECTRA - MOLECULAR ION DETECTION =====

def get_all_peaks(df_sample):
    """
    Taking all m/z values and their intensities select all
    m/z values greater than some threshold.
    Sample should be processed.
    """
    # get max for each m/z ion
    df_mra_mz = df_sample.groupby('m/z')['abun_scaled'].agg('max')
    
    # Compute threshold as 1% of the maximum abundance in the sample
    abun_thrs = np.round(df_mra_mz.max()) * config.INTENSITY_THRESHOLD
    
    # Select only m/z ions with max abundance greater than the threshold
    df_mra_mz = df_mra_mz[df_mra_mz > abun_thrs]

    return df_mra_mz
    

def get_molecular_ion(df_sample):
    """
    Find molecular ion from the sample.
    Sample should be processed.
    """
    
    # Get all peaks above a threshold
    df_mra_mz = get_all_peaks(df_sample)
    
    # Select the ion with the greatest mass
    df_mra_mz = df_mra_mz.reset_index()
    df_mra_mz = df_mra_mz[df_mra_mz['m/z'] == df_mra_mz['m/z'].max()]
    
    M = df_mra_mz.iloc[0]['m/z']
    M_mra = df_mra_mz.iloc[0]['abun_scaled']
    
    return M, M_mra


def get_Mp1_peak(df_sample):
    """
    Check whether there is an [M+1] peak and record the m/z value and the 
    mra.
    Sample should be processed.
    """
    
    M, M_mra = get_molecular_ion(df_sample)
    
    Mp1 = M+1
    
    Mp1_mra = df_sample[df_sample['m/z'] == Mp1]['abun_scaled'].max()
    
    Mp1_isotope = (Mp1_mra/M_mra)*100
    
    return Mp1_mra, Mp1_isotope
   

def get_Mp2_peak(df_sample):
    """
    Check whether there is an [M+2] peak and record the m/z value and the
    mra.
    Sample should be processed.
    """
    M, M_mra = get_molecular_ion(df_sample)
    Mp2 = M+2
    Mp2_mra = df_sample[df_sample['m/z'] == Mp2]['abun_scaled'].max()
    Mp2_isotope = (Mp2_mra/M_mra)*100
    
    return Mp2_mra, Mp2_isotope
    

def get_no_carbons(df_sample):
    """
    Compute number of carbons.
    Sample should be processed.
    """
    _, M_mra = get_molecular_ion(df_sample)
    Mp1_mra, _ = get_Mp1_peak(df_sample)

    return np.round((Mp1_mra/M_mra) * (100/1.1),2)
    
   
def get_moli_frag_peak_diff(df_sample):
    """
    Compute the difference in mass from the molecular peak 
    to the first closest fragment.
    Sample should be processed.
    """
    # Compute all peaks
    df_sample_all_peaks = get_all_peaks(df_sample)
    
    mz_values = df_sample_all_peaks.reset_index().tail(2)['m/z'].values
    
    return mz_values[1] - mz_values[0]
    
     
    
def features_ms(metadata, file_list, detrend_method):
    """Features from mass spectra.
    """
    df = pd.DataFrame()
    
    for file_idx in tqdm(file_list): #idx
        #print(file_idx)
        temp = {}
        
        # Select a sample and get sample name
        df_sample = preprocess.get_sample(metadata, file_idx)
        sample_name = metadata.iloc[file_idx]['sample_id']
        temp['sample_id'] = sample_name
        
        # Preprocess the sample
        df_sample = preprocess.preprocess_samples(df_sample, 
                                                detrend_method=detrend_method)
        
        # Molecular ion and its abundance
        M, M_mra = get_molecular_ion(df_sample)
        temp['M'] = M
        temp['M_mra'] = M_mra 
        temp = pd.DataFrame(data=temp, index=[0])
        
        # Isotopic abundance [M+1]
        Mp1_mra, Mp1_isotope = get_Mp1_peak(df_sample)
        temp['Mp1_mra'] = Mp1_mra
        temp['Mp1_isotope'] = np.round(Mp1_isotope,2)
        
        # Isotopic abundance [M+2]
        Mp2_mra, Mp2_isotope = get_Mp2_peak(df_sample)
        temp['Mp2_mra'] = Mp2_mra
        temp['Mp2_isotope'] = np.round(Mp2_isotope,2)
        
        
        # 1st heavy isotopes
        if np.round(Mp1_isotope,3) in pd.Interval(0.014,0.016, closed='both'): temp['hi1'] = 1. #'H'
        elif np.round(Mp1_isotope,1) in pd.Interval(1.0,1.2, closed='both'): temp['hi1'] =  12. #'C'
        elif np.round(Mp1_isotope,2) in pd.Interval(0.36,0.38, closed='both'): temp['hi1'] =  14. #'N'
        elif np.round(Mp1_isotope,2) in pd.Interval(0.03,0.05, closed='both'): temp['hi1'] =  16. #'O'
        elif np.round(Mp1_isotope,1) in pd.Interval(5.0,5.2, closed='both'): temp['hi1'] =  28. #'Si'
        elif np.round(Mp1_isotope,1) in pd.Interval(0.7,0.9, closed='both'): temp['hi1'] =  32. #'S'
        else: temp['hi1'] = ''
        
        # 2nd heavy isotopes
        if np.round(Mp2_isotope,1) in pd.Interval(0.1, 0.3, closed='both'): temp['hi2'] = 16. #'O'
        elif np.round(Mp2_isotope,1) in pd.Interval(3.3, 3.5, closed='both'): temp['hi2'] =  28. #'Si'
        elif np.round(Mp2_isotope,1) in pd.Interval(4.3, 4.5, closed='both'): temp['hi2'] =  32. #'S'
        elif np.round(Mp2_isotope,1) in pd.Interval(32.4,32.6, closed='both'): temp['hi2'] =  35.5 #'Cl'
        elif np.round(Mp2_isotope,1) in pd.Interval(97.9,98.1, closed='both'): temp['hi2'] =  79.9 #'Br'
        else: temp['hi2'] = ''
        
        #TODO Adjusted M+1 and M+2
        
        # Number of carbons
        #TODO Needs correction if there are present 1st and 2nd isotopes
        temp['C_cnt'] = get_no_carbons(df_sample)
        
        
        # Presence of nitrogen
        # an even molecular ion indicates the sample lacks nitrogen
        #N_cnt = nitrogen_rule(df_sample)
        #temp['N_cnt'] = N_cnt
            
        # Difference between molecular ion and its first fragment peak
        mi_frg_diff = get_moli_frag_peak_diff(df_sample)
        temp['mi_frg_diff'] = mi_frg_diff
        
        
        
            
        df = pd.concat([df, temp], axis=0)
    
    return df.reset_index(drop=True)
        
        
# ===== MZ STATISTICS =====

def get_mz_stats(df_sample, sample_name, smooth):
    
    if smooth:
        sample_stats = df_sample.groupby('m/z')\
            .agg({'abun_scaled_smooth': ['mean', 'median', 'std',
                                pd.DataFrame.kurtosis,
                                pd.DataFrame.skew]})
    else:
        sample_stats = df_sample.groupby('m/z')\
            .agg({'abun_scaled': ['mean', 'median', 'std',
                                pd.DataFrame.kurtosis,
                                pd.DataFrame.skew]})
    sample_stats = sample_stats.droplevel(level=0, axis=1).reset_index()
    sample_stats['sample_id'] = sample_name
    
    return sample_stats


def features_mz_stats(metadata, file_list, detrend_method):
    """
    Statistics for each m/z.
    """
    df = pd.DataFrame()
    
    for file_idx in tqdm(file_list): #idx
        #print(file_idx)
        temp = {}
        
        # Select a sample and get sample name
        df_sample = preprocess.get_sample(metadata, file_idx)
        sample_name = metadata.iloc[file_idx]['sample_id']
        temp['sample_id'] = sample_name
        
        # Preprocess the sample
        df_sample = preprocess.preprocess_samples(df_sample,
                                                detrend_method=detrend_method)
        
        # Statistics
        sample_stats = get_mz_stats(df_sample, sample_name)
        
        # Prepare the df
        stats_pivot = sample_stats.pivot(index='sample_id', columns='m/z')
        df = pd.concat([df, stats_pivot], axis=0)
    
    df = df.fillna(0)
    df.columns = df.columns.map(lambda x: '_'.join([str(i) for i in x]))
    
    return df

      
# ===== FIND PEAKS =====

def get_reference_peak(df_sample):
    """
    Find the ion with the peak with the max abundance in the sample.    
    Sample should be preprocessed.
    
    Which ions mass/charge represent a reference peak in the sample.
    1. Find peaks for each ion m/z
    2. Get the max abundance of each ion m/z in case there are more than one
    3. Get the ion with the max abundance
    4. Return that ion m/z
    """
    
    max_peak_ion = 0
    peak_abund = 0
    
    # Get the list of ions present in the sample
    ion_list = list(df_sample['m/z'].unique())
    
    for ion in ion_list: 
        # Select data only for one ion m/z
        temp_dt = df_sample[df_sample['m/z'] == ion].copy()
        
        # Apply Gaussian filter for the values
        temp_dt['abun_minsub_scaled_filtered'] = gaussian_filter1d(temp_dt['abun_scaled'], 
                                                                sigma=4)
        
        # Compute the median for the prominence ("the minimum height necessary 
        # to descend to get from the summit to any higher terrain")
        med = temp_dt['abun_minsub_scaled_filtered'].median()
        
        # Find peaks - return index of the peak
        peaks, _ = find_peaks(temp_dt['abun_minsub_scaled_filtered'], 
                              prominence=med)

        if len(peaks) > 0:
            for peak in peaks:
                  ma = temp_dt.iloc[peak]['abun_scaled']
                  if ma > peak_abund:
                      peak_abund = ma
                      max_peak_ion = ion
                      
    return max_peak_ion


def compute_ion_peaks(metadata, 
                      sample_idx, 
                      detrend_method, 
                      gauss_sigma:int=5):
    
    # Select a sample and get sample name
    df_sample = preprocess.get_sample(metadata, sample_idx)
    sample_name = metadata.iloc[sample_idx]['sample_id']

    # Preprocess the sample
    df_sample = preprocess.preprocess_samples(df_sample,
                                              detrend_method=detrend_method,
                                              )
    mz_list = df_sample['m/z'].unique().tolist()    
    # Compute stats and save in dict for each ion type
    ion_peaks_cnt = {} # Initialize dictionary to save calculated values
    for ion in mz_list:
        #print(colored(f'ION: {ion}','blue'))
        ion_peaks_info = [] # initialize list to store stats per ion type
        
        temp_dt = df_sample[df_sample['m/z'] == ion].copy()
        
        # Apply Gaussian filter for the values
        temp_dt['abun_minsub_scaled_filtered'] = gaussian_filter1d(temp_dt['abun_scaled'],
                                                                sigma=gauss_sigma)
        
        # Compute the median for the prominence ("the minimum height necessary
        # to descend to get from the summit to any higher terrain")
        med = temp_dt['abun_minsub_scaled_filtered'].median()
        
        # Find peaks
        peaks, _ = find_peaks(temp_dt['abun_minsub_scaled_filtered'],
                              prominence=med)
        ion_peaks_info.append(len(peaks))
        #if len(peaks) > 0: print(f'\nIon: {ion}, Peaks: {peaks}')
        
        # Peak statistics
        #TODO Add distance from peaks
        peak_temp = []
        peak_time = []
        peak_abund = []
        for i in peaks:
            tm = temp_dt.iloc[i]['time']; peak_time.append(tm) 
            t = temp_dt.iloc[i]['temp']; peak_temp.append(t)
            a = temp_dt.iloc[i]['abun_scaled']; peak_abund.append(a)
            #if len(peaks) > 0: print(f'Peak {i} Abund: {a}')
            
        if len(peak_time)>0 and len(peak_temp)>0 and len(peak_abund)>0:
            peak_time = max(peak_time)
            peak_temp = max(peak_temp)
            peak_abund = max(peak_abund)
        else: 
            peak_time, peak_temp, peak_abund = 0, 0, 0
            
        # Compute AUC
        #TODO Needs further discussion
        #if not temp_dt.empty:
        #    area_abund = np.round(auc(temp_dt['temp'],temp_dt['abun_scaled']),5)
        #else: area_abund = 0
        
        # Add values
        ion_peaks_info.append(peak_time)
        ion_peaks_info.append(peak_temp)
        ion_peaks_info.append(peak_abund)
        #ion_peaks_info.append(area_abund)
            
        ion_peaks_cnt[ion] = ion_peaks_info
        
    #-------------------
    # CONVERT DICT TO DF
    # Define new column names
    new_cols = ['m/z','peak_cnt', 'peak_time', 'peak_temp', 'peak_abund']
    
    # Save dict as data frame and transpose
    ion_peaks_stats = pd.DataFrame(ion_peaks_cnt, dtype='float64')
    ion_peaks_stats = ion_peaks_stats.T
    ion_peaks_stats.reset_index(inplace=True)
    ion_peaks_stats.columns = new_cols
    ion_peaks_stats['sample_id'] = sample_name
    
    #-----------------------------------------
    # CREATE PIVOT FROM DF FOR FEATURE COLUMNS
    df = ion_peaks_stats.pivot(index='sample_id', columns="m/z")
    
    return df


# Concat all samples together
# Loop over all files compute peaks and concat together
def features_ion_peaks(file_paths:dict, metadata, detrend_method):
    """
    Combines all computed ion peaks stats from each sample
    into a features data frame.
    """
    # Initialize a data frame to store all the sample calculations
    df = pd.DataFrame()
    
    for sample_idx in tqdm(file_paths):
        ion_peaks_df = compute_ion_peaks(metadata, sample_idx, detrend_method)
        df = pd.concat([df,ion_peaks_df], axis = 0)
    
    # Join multi column index into one separated by -
    df.columns = df.columns.map(lambda x: '_'.join([str(i) for i in x]))
    
    return df

# ===== AREA - SAMPLE =====

def sample_abund_area(metadata, idx, detrend_method):
    """
    Compute the area under the abun_scaled for
    the whole sample given time.
    """
    df_sample = preprocess.get_sample(metadata, idx)
    df_sample = preprocess.preprocess_samples(df_sample, detrend_method=detrend_method)
    df_sample = df_sample.sort_values(by=['time', 'abun_scaled'])

    # Sum all mz values per time to get a sample mz value per time
    df_sample['abun_sum_time'] = df_sample.groupby('time')['abun_scaled'].transform('sum')
    df_sample = df_sample[['time', 'abun_sum_time']]\
                    .drop_duplicates()\
                    .sort_values(by=['time', 'abun_sum_time'])\
                    .copy()

    x = df_sample['time'].values
    y = df_sample['abun_sum_time'].values
    area = np.trapz(y=y,x=x)

    return area


# fts_area_sample
def features_area(files, metadata, detrend_method):
    
    areas_dict = {}
    
    for idx in tqdm(files):
        sample_name = metadata.iloc[idx]['sample_id']
        area_sample = sample_abund_area(metadata, idx, detrend_method)
        areas_dict[idx] = area_sample
        
    return areas_dict


# ===== AREA - CONTRIBUTION PER TEMP BIN =====

def area_contrib_temp(metadata, idx, detrend_method):
    #TODO FInish - the values look a bit weird
    # Read in the sample
    df_sample = preprocess.get_sample(metadata, idx)
    df_sample = preprocess.preprocess_samples(df_sample, detrend_method)

    # COmpute area for the sample
    sample_area = sample_abund_area(metadata, idx, detrend_method)
    
    # Define the temperature bins
    temprange = pd.interval_range(start=-100, end=1500, freq=100)
    df_sample['temp_bin'] = pd.cut(df_sample['temp'], bins=temprange)
    
    # Compute areas per temp bin
    temp_bin_areas_dict = {}
    temp_bin_list = df_sample['temp_bin'].unique().tolist()
    for bin in temp_bin_list:
        df_temp = df_sample[df_sample.temp_bin == bin][['time', 'abun_scaled']].copy()
        df_temp = df_temp.sort_values(by='time')
        x = df_temp['time'].values
        y = df_temp['abun_scaled'].values
        temp_bin_areas_dict[bin] = (np.trapz(y=y,x=x) / sample_area) * 100
    
    return temp_bin_areas_dict




# ===== ABUNDANCE TO TEMP =====

def range_abun_to_temp(metadata, idx, detrend_method):
    """
    Compute ratio of abundance value to temperature.
    Divide temp into bins and for each bin and m/z value
    compute range of ratios of abundance to temperature.
    The goal is to see how some m/z values are affected by 
    temp value.
    """
    # Read in the sample
    df_sample = preprocess.get_sample(metadata, idx)
    df_sample = preprocess.preprocess_samples(df_sample, detrend_method)
    sample_name = metadata.iloc[idx]['sample_id']

    temprange = pd.interval_range(start=-100, end=1500, freq=100)
    df_sample['temp_bin'] = pd.cut(df_sample['temp'], bins=temprange)

    # Abundance as a function of temperature
    df_sample['abun_temp_perc'] = df_sample['abun_scaled'] / df_sample['temp']

    # Compute the range for each m/z
    df_sample['abun_to_temp_range'] = df_sample\
        .groupby(['temp_bin', 'm/z'])['abun_temp_perc']\
        .transform(np.ptp)
    df_sample = df_sample[['temp_bin','m/z','abun_to_temp_range']].drop_duplicates().reset_index(drop=True)
    df_sample['sample_id'] = sample_name

    # Fix m/z adn temp-bin names 
    df_sample['m/z'] = ['mz_'+str(i) for i in df_sample['m/z']]
    df_sample['m/z'] = [i.removesuffix('.0') for i in df_sample['m/z']]
    df_sample['temp_bin'] = ['temp_'+str(i) for i in df_sample['temp_bin']]
    
    # Turn into a pivot where columns are combo of temp_bin & m/z
    # and values are abun_to_temp_range
    df_sample_pivot = df_sample.pivot(columns=['temp_bin', 'm/z'], 
                                      values='abun_to_temp_range',
                                      index='sample_id')
    # Join multiple column indices
    df_sample_pivot.columns = df_sample_pivot.columns.map(lambda x: '_'.join([str(i) for i in x]))
    # Rename columns
    t_cols = df_sample_pivot.columns
    remove_chars = "(,]"
    for char in remove_chars:
        t_cols = [i.replace(char,'') for i in t_cols]
    t_cols = [i.replace(' ','_') for i in t_cols]
    df_sample_pivot.columns = t_cols
    # Add prefix to column names
    df_sample_pivot = df_sample_pivot.add_prefix('r_abun_temp_')

    return df_sample_pivot



# ===== DEEP LEARNING =====
#TODO Add more features and run again
#TODO Add to the final ensemble - check maybe some of the targets will benefit
def dl_time_pivot(metadata, n_sample, max_time):
    """
    Process the time series of a sample to create a df
    where each row is a  distinct time. Columns represent
    features given the m/z value.
    """
    
    # Data frame to store the final processed - all samples (by row)
    df = pd.DataFrame()
    
    # ----- SAMPLE PROCESSING -----
    # Load and preprocess the sample
    df_sample = preprocess.get_sample(metadata, n_sample)
    df_sample = preprocess.preprocess_samples(df_sample)
    
    # Get sample name and instrument
    sample_name = metadata.iloc[n_sample]['sample_id']
    instrument = metadata.iloc[n_sample]['instrument_type']
    
    # Define the time range
    time_range = pd.interval_range(start=0.0, 
                               end=utils.roundup(max_time), 
                               freq=10, 
                               closed='left')
    
    # Map the time into bins
    df_sample['time_bin'] = pd.cut(df_sample['time'], bins=time_range)
    del df_sample['time']
    
    # Aggregate features based on the time_bin, temp and m/z
    # Solves the problem of several measurement within the interval range
    # Aggregate temp and abundance by mean on time_bin and m/z
    df_sample_agg = df_sample.groupby(['m/z', 'time_bin']).agg('mean').reset_index()
    
    # There are still duplicates in temp in the sam-testbed samples
    # Compute standard deviation and store as variable and take the average
    # of temp for final value
    df_sample_agg['temp_osc_time'] = df_sample_agg.groupby('time_bin')['temp'].transform('std')
    df_sample_agg['temp'] = df_sample_agg.groupby('time_bin')['temp'].transform('mean')
    
    # Make a pivot table
    df_pivot = df_sample_agg.pivot(index=['time_bin', 'temp', 'temp_osc_time'],
                               columns='m/z', 
                               values='abun_scaled')
    
    df_pivot = df_pivot.add_prefix('mz_')
    df_pivot.columns = [i.removesuffix('.0') for i in df_pivot.columns]
    df_pivot = df_pivot.add_suffix('_abund')
    df_pivot['sample_id'] = sample_name
    df_pivot['instrument_type'] = instrument
    
    df_pivot = df_pivot.reset_index()
    
    return df_pivot
    
def dl_ts(metadata, max_time):
    """
    Create a 3D array of time series for DL models.
    1D - samples
    2D - features
    3D - time step
    """
    
    df = pd.DataFrame()
    
    for i in tqdm(range(metadata.shape[0])):
        df_pivot = dl_time_pivot(metadata, i, max_time)
        df = pd.concat([df, df_pivot], axis=0)
    
    return df
    
      

# Compute average abundance for each sample
def compute_abundance_per_ion(df_sample, sample_name:str, stat:str):
    """ Compute average abundance per ion type"""
    dt = pd.DataFrame(df_sample.groupby('m/z')['abundance'].agg(stat)).reset_index()
    dt['sample_id'] = sample_name
    dt.columns = ['Ion_type', 'avg_abundance', 'sample_id']
    
    return dt

# Features are ions
def ion_abundance(df_meta, stats:str, sample_list:list):
    """Compute avergae abundance for each ion in the sample
    """
    
    # Initialize a table to store computed values
    dt = pd.DataFrame(dtype='float64')
    
    
    # Loop over all sample_id and compute. Add computation to dt.
    print(f'Number of samples: {len(sample_list)}')
    
    for i, sample in enumerate(sample_list):
        
        df_sample = preprocess.get_sample(df_meta, i)
        
        # Preprocess data sample
        df_sample = preprocess.preprocess_samples(df_sample)
        
        # Compute average abundance for each sample
        if stats == 'average':
            dt_avg_abund = compute_abundance_per_ion(df_sample, 
                                                     sample_name=sample)
        elif stats == 'max':
            dt_avg_abund = compute_abundance_per_ion(df_sample, 
                                                     sample_name=sample)
            
        # Add computed values back to original data frame
        dt = pd.concat([dt, dt_avg_abund], ignore_index=True)
        
    return dt


def avg_temp_sample(df, df_files):
    # Average temperature per sample
    avg_temp_sample = {}
    for i in range(len(df_files)):
        df_sample = pd.read_csv(config.DATA_DIR + df_files[i])
        avg_temp_sample[i] = df_sample.temp.mean()
        #print(avg_temp_sample[i])
        
    df['avg_temp'] = df.index.map(avg_temp_sample)
    
    return df

def slope_time_temp(train_files:dict, metadata, detrend_method):
    """
    Compute slope of time ~ temp for each sample using
    linear regression. Is there any difference between
    commercial and sam_testbed samples?
    """

    coefs_lr = {}

    for i in tqdm(train_files):
        ht = preprocess.get_sample(metadata, i)
        ht = preprocess.preprocess_samples(ht, detrend_method=detrend_method)
        sample_name = metadata.iloc[i]['sample_id']

        lr = LinearRegression()

        X = np.array(ht['time']).reshape(-1, 1)
        y = ht['temp'].values
        lr.fit(X, y)

        coefs_lr[sample_name] = lr.coef_[0]

    return coefs_lr




# ===== CORRELATION WITH MZ4 =====

def corr_mz4(df_sample,
             sample_name:str,
             corr_method:str='spearman'):
    """
    Correlation of mz=4 with other mz values.
    """
    mz_list = df_sample['m/z'].unique().tolist()
    
    if 4.0 in mz_list:
        # Create a pivot table so that each m/z value is a column
        df_sample = df_sample.pivot(index='time',
                                    columns='m/z',
                                    values='abun_scaled')

        # Compute correlation
        df_corr = df_sample.corr(method=corr_method)
        # Remove m/z 4 from index and select only mz4 from the columns
        # to get mz4 correlation with all other mz values
        df_corr = df_corr[df_corr.index != 4.0][[4.0]]
        # Reset index to get mz as a column
        df_corr = df_corr.reset_index()
        df_corr.columns = ['m/z', 'mz_4']
        
        # Fix column names
        df_corr['m/z'] = [str(i).removesuffix(".0") for i in df_corr['m/z']]
        df_corr['m/z'] = ['mz_' + str(i) for i in df_corr['m/z']]
        
    else:
        df_corr = pd.DataFrame(index=mz_list)
        df_corr = df_corr.reset_index()
        df_corr.columns = ['m/z']
        df_corr['mz_4'] = np.nan
    
    # Add sample_id
    df_corr['sample_id'] = sample_name

    # Create a data frame where m/z values are columns
    df_corr_pivot = df_corr.pivot(index='sample_id',
                                columns='m/z',
                                values='mz_4')

    return df_corr_pivot


def lr_corr_mz4(df_sample, 
                corr_method:str='spearman'):
    """
    Linear regression of the 
    correlation coefficient between 
    m/z=4 and all other m/z values.
    """
    mz_list = df_sample['m/z'].unique().tolist()

    if 4.0 in mz_list:
        # Create a pivot table so that each m/z value is a column
        df_sample = df_sample.pivot(index='time',
                                    columns='m/z',
                                    values='abun_scaled')

        # Compute correlation
        df_corr = df_sample.corr(method=corr_method)
        # Remove m/z 4 from index and select only mz4 from the columns
        # to get mz4 correlation with all other mz values
        df_corr = df_corr[df_corr.index != 4.0][[4.0]]
        # Reset index to get mz as a column
        df_corr = df_corr.reset_index()
        df_corr.columns = ['m/z', 'mz_4']

        # Remove all mz values with corr less than the threshold
        tmp = df_corr[np.abs(df_corr['mz_4']) > config.CORRELATION_THRESHOLD]
    else:
        tmp = pd.DataFrame()

    # Compute linear regression
    if not tmp.empty:
        lr = LinearRegression()
        X = np.array(tmp['m/z']).reshape(-1,1)
        y = tmp['mz_4'].values
        lr.fit(X,y)
        lr_coef = lr.coef_[0]
    else:
        lr_coef = np.nan

    return lr_coef



# ===== TEMPERATURE QUANTILE =====

def tempIQ_mzstats(df_sample,
                   no_quantiles):

    df_sample['tempIQ'] = pd.qcut(df_sample['temp'],
                                  q=no_quantiles,
                                  precision=0)
    
    iq_stats = df_sample.groupby(['tempIQ', 'm/z'])\
                        .agg({'time': ['max', np.ptp],
                              'temp': ['max', np.ptp],
                              'abun_scaled': ['mean','max', 'std', 'var', np.ptp]})\
                        .fillna(0)
                        
    return iq_stats
    


def features_temp_qcut(df_all, 
                       split:str='train',
                       no_quantiles:int=10, 
                       precision:int=0,
                       tr_bins=None):
    """
    Compute defined measures given temperature bin.
    The temperature bin is computed using qcut with
    q=10, i.e. the deciles. The binning is calculated
    for each m/z ratio.
    All the sample should be preprocessed.
    
    df_all (pandas data frame) data frame of all time series
            data stacked in rows.
    split (string; 'train','valid'; default='train')
            
    """
    #TODO Nede to write a sklearn fit_transform class to use transform
    # for the validation & test set
    
    if split in ['train']:
        tempD, tr_bins = pd.qcut(df_all['temp'], 
                                    q=no_quantiles, 
                                    precision=precision,
                                    retbins=True)
        df_all['TEMP_D'] = tempD
    else:
        df_all['TEMP_D'] = pd.cut(df_all["temp"], 
                                  bins=tr_bins, 
                                  include_lowest=True)
        
    # Compute features
    ion_stats = df_all.groupby(['sample_id', 'm/z', 'TEMP_D'])\
                            .agg({'time': ['max', np.ptp],
                                'temp': ['max', np.ptp],
                                'abun_scaled': ['mean', 'max', np.ptp]})\
                            .fillna(0)
    #TODO Add slope of the abun and temp curve for the bin
        
    
    ion_stats = ion_stats.reset_index()
        
    df_fts_pivot = ion_stats.pivot(index='sample_id', columns=['TEMP_D', 'm/z'])
    df_fts_pivot.columns = df_fts_pivot.columns.map(lambda x: '_'.join([str(i) for i in x]))
    # Rename columns
    t_cols = df_fts_pivot.columns
    remove_chars = "(,]"
    for char in remove_chars:
        t_cols = [i.replace(char,'') for i in t_cols]
    t_cols = [i.replace(' ','_') for i in t_cols]
    df_fts_pivot.columns = t_cols
    
    if split in ['train']:
        return df_fts_pivot, tr_bins
    else:
        return df_fts_pivot




# ===== TARGET ENCODING =====
# Target encode each label on instrument type and save
# as a variable. There should be 11 additional variables
def label_encode(df, 
                 feature:str, 
                 target:str, 
                 min_samples_leaf=1,
                 smoothing=1):
    """
    Target encode feature. Feature and label should be in
    the same data frame.
    """
    # Compute target mean and count
    averages = df.groupby(feature)[target].agg(['mean', 'count'])
    
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    
    prior = df[target].mean()
    averages[target] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    #averages = averages.reset_index().pivot(columns='instrument_type', 
    #                                        index='sample_id')
    #averages.columns = averages.columns.map(lambda x: '_'.join([str(i) for i in x]))
    
    return averages


def label_encode_multi(df, df_test, feature, target_labels_list:str):
    #TODO Fix to work on CV
    le_dict = {}
    
    for label in target_labels_list:
        le = label_encode(df, feature, label)
        le_dict = le_dict | le.to_dict()
        
        df['le_'+label] = df[feature].map(le.to_dict()[label])
        df_test['le_'+label] = df_test[feature].map(le.to_dict()[label])
        
    df = df.drop(target_labels_list + [feature], axis=1)    
    df_test = df_test.drop([feature], axis=1)    
    
    return df, df_test, le_dict
    

def get_topN_ions(metadata, N:int=3, normalize:bool=True, 
                  lb:float=0.0, ub:float=0.99):
    """
    Compute top N ions by their max relative
    abundance. 
    
    Parameters  
    ----------
        metadata: pandas data frame
        
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
    for i in tqdm(range(metadata.shape[0])):
        # Get and preprocess data sample
        hts = preprocess.get_sample(metadata, i)
        hts = preprocess.preprocess_samples(hts)
        sample_name = metadata.iloc[i]['sample_id']
        
        # Compute top 3 ions by relative abundance
        # Take max of each ion group sort and slice top N
        top3 = list((hts.groupby('m/z')['abun_scaled']\
                        .agg('max')\
                        .sort_values(ascending=False))\
                            .head(N).index)
        
        top3_ions[sample_name] = top3
    
    # Convert to data frame
    temp = pd.DataFrame.from_dict(top3_ions, orient='index')
    # Rename columns
    temp.columns = ['top_%s' % (i+1) for i in range(N)]
    
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
    
    return temp


# ===== CORRELATION =====


def corr_peak_mz(df_sample):
    """
    Compute nonparametric correlation between the
    peak and the rest of the m/z values.
    
    df_sample: pandas data frame
        The sample should be processed!
    """
    
    # Check that the length of all ion time series is equal
    #TODO sample 401 does not have equal time periods
    #if len(df_sample.groupby('m/z')['time'].agg('count').unique()) != 1:
    #    print(colored(f'Sample {sample_name} has irregular time intervals across samples.','red'))

    # Get the ion list
    sample_ions = list(df_sample['m/z'].unique())
    #print(f'Number of ions: {len(sample_ions)}')
    
    # Correlation df
    df_corr = pd.DataFrame(index=sample_ions)
    
    # Get the ion with the peak of max abundance
    ref_peak = get_reference_peak(df_sample)
    assert len([ref_peak]) == 1
    
    if ref_peak > 0:
        for peak in [ref_peak]:
            ion_i = df_sample[df_sample['m/z'] == peak]['abun_scaled'].values
        
            for j in sample_ions:
                ion_j = df_sample[df_sample['m/z'] == j]['abun_scaled'].values
                
                # Values mesured in same time intervals for two ions
                all(df_sample[df_sample['m/z'] == peak]['time'].values == df_sample[df_sample['m/z'] == j]['time'].values)

                sprcorr, _ = spearmanr(ion_i, ion_j)
                df_corr.loc[j,'Ion_' + str(peak)] = sprcorr
    
    # Select ions with significant correlation
    suffix = ".0"
    df_corr.columns = [str(i).removesuffix(suffix) for i in df_corr]
    df_corr.index = [str(i).removesuffix(suffix) for i in df_corr.index]
    df_corr_sig = df_corr[df_corr.apply(lambda x: np.abs(x) > 0.6, 
                                        axis=1) == True].dropna()
    
    # Return a list of selected ions
    corr_ions = df_corr_sig.index.astype('float')

    return corr_ions.tolist()


def corr_ions_sig(metadata):
    """
    Determine a list of ions with high correlation
    within each sample. Combine list from all samples
    to create a list for the training set to take into
    account.
    """
    
    # Get the dict of all file paths: TR, VL, TE
    files_list = metadata['features_path'].to_dict()
    
    # Store correlated ions
    ions_corr = []
    
    for i in tqdm(files_list):
        
        # Get and preprocess sample
        df_sample = preprocess.get_sample(metadata,i)
        df_sample = preprocess.preprocess_samples(df_sample)
        #print(metadata.iloc[i]['sample_id'])
        
        # Compute significantly correlated ions
        corr_ions = corr_peak_mz(df_sample)
        ions_corr = list(set(ions_corr) | set(corr_ions))   # Union of two sets
        #print(ions_corr)
        
    return ions_corr


# ===== WIDTH =====

def find_roots(x, y):
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return np.round(x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1),2)

def plot_peaks_mz(metadata, idx, mz):
    df_sample = preprocess.get_sample(metadata,idx)
    df_sample = preprocess.preprocess_samples(df_sample, 
                                              detrend_method='min')

    # Select the mz ratio and sort values by temperature
    df_mz = df_sample[df_sample['m/z'] == mz].copy()

    # Apply smoothing to the abundance values
    df_mz['abun_scaled_smooth'] = gaussian_filter1d(df_mz['abun_scaled'], 
                                                    sigma=4, 
                                                    order=0)
    mz_prominence = df_mz['abun_scaled_smooth'].median()
    print(f'Prominence: {mz_prominence}')
    
    x = df_mz['temp'].values
    y = df_mz['abun_scaled_smooth'].values
            
    peaks, _ = find_peaks(df_mz['abun_scaled_smooth'], 
                        prominence=mz_prominence)
    print(f'Peak idx: {peaks}')
    
    # Compute only for those mz ratios which have at least
    # one peak.
    if len(peaks) > 0:
        # Get temp values for peaks
        peaks_temp = []
        for i in peaks:
            peaks_temp.append(df_mz.iloc[i]['temp'])
        print(f'Peak temp: {peaks_temp}')
        
        # At which MRA should the width be calculated
        width_loc = [0.25, 0.5, 0.75]
        peaks_mra_width_loc = {}
        
        for n,p in enumerate(peaks): 
            print(f'Peak {p}')
            mra_width_loc = []
            # Value of abundance at peak
            w = df_mz.iloc[p]['abun_scaled_smooth']
            
            # Compute width at three points
            for i in width_loc:
                width_point_mra = w*i
                #mra_width_loc.append(width_point_mra)
            
                # Compute roots
                roots = find_roots(x, y-width_point_mra)            
                print(i, roots)
                # Select only roots for the corresponding peak
                # Each peak should have 2 measurements, this ensures there
                # are no curves in between the peaks.
                if len(peaks)*2 == len(roots):
                    width_range = roots[n:(n+2)]
                    #mra_width_loc.append(width_range)
                    peaks_mra_width_loc[p] = width_range

    return peaks_mra_width_loc
