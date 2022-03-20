"""Feature engineering"""


from importlib.metadata import metadata
from operator import index
import pandas as pd
import numpy as np
from src import config, preprocess, utils
from scipy.signal import find_peaks
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import auc
from tqdm import tqdm
import gc


# BENCHMARK FEATURES
# Bin the temp and compute processes max abundance for each bin
def bin_temp_abund(df_sample, sample_name:str):
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
    df_sample = preprocess.preprocess_samples(df_sample)
    
    # Compute max relative abundance
    ht = df_sample.groupby(['m/z', 'temp_bin'])['abun_minsub_scaled'].agg('max').reset_index()
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


def features_iontemp_abun(df_meta, sample_list):
    
    # Initialize a table to store computed values
    dt = pd.DataFrame(dtype='float64')
    
    #ion_temp_dict = {}
    
    # Loop over all sample_id and compute. Add computation to dt.
    print(f'Number of samples: {len(sample_list)}')
    for i in sample_list:
        sample_name = df_meta.iloc[i]['sample_id']
        #print(sample_name)
        df_sample = preprocess.get_sample(df_meta, i)
        
        ht_pivot = bin_temp_abund(df_sample, sample_name)
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
     

# ===== Duration to max temperature per ion =====
def ion_duration_maxtemp(df_sample, ion_list):
    
    # Preprocess the sample data
    df_sample = preprocess.preprocess_samples(df_sample)

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
        

# ===== FIND PEAKS =====
def compute_ion_peaks(metadata, sample_idx, ion_list):
    
    # Select a sample and get sample name
    df_sample = preprocess.get_sample(metadata, sample_idx)
    sample_name = metadata.iloc[sample_idx]['sample_id']
    
    # Preprocess the sample
    df_sample = preprocess.preprocess_samples(df_sample)
    
    # Compute stats and save in dict for each ion type
    ion_peaks_cnt = {} # Initialize dictionary to save calculated values
    for ion in ion_list:
        ion_peaks_info = [] # initialize list to store stats per ion type
        
        temp_dt = df_sample[df_sample['m/z'] == ion].copy()
        
        # Apply Gaussian filter for the values
        temp_dt['abun_minsub_scaled_filtered'] = gaussian_filter1d(temp_dt['abun_minsub_scaled'], 
                                                                sigma=4)
        
        # Compute the median for the prominence ("the minimum height necessary 
        # to descend to get from the summit to any higher terrain")
        med = temp_dt['abun_minsub_scaled_filtered'].median()
        
        # Find peaks
        peaks, _ = find_peaks(temp_dt['abun_minsub_scaled_filtered'], prominence=med)
        ion_peaks_info.append(len(peaks))
        
        # Peak statistics
        peak_temp = []
        peak_time = []
        peak_abund = []
        for i in peaks:
            tm = temp_dt.iloc[i]['time']; peak_time.append(tm) 
            t = temp_dt.iloc[i]['temp']; peak_temp.append(t)
            a = temp_dt.iloc[i]['abun_minsub_scaled']; peak_abund.append(a)
        
        if len(peak_time)>0 and len(peak_temp)>0 and len(peak_abund)>0:
            peak_time = max(peak_time)
            peak_temp = max(peak_temp)
            peak_abund = max(peak_abund)
        else: 
            peak_time, peak_temp, peak_abund = 0, 0, 0
            
        # Compute AUC
        #TODO Needs further discussion
        #if not temp_dt.empty:
        #    area_abund = np.round(auc(temp_dt['temp'],temp_dt['abun_minsub_scaled']),5)
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
def features_ion_peaks(file_paths:dict, metadata, ion_list:list):
    """
    Combines all computed ion peaks stats from each sample
    into a features data frame.
    """
    # Initialize a data frame to store all the sample calculations
    df = pd.DataFrame()
    
    for sample_idx in tqdm(file_paths):
        ion_peaks_df = compute_ion_peaks(metadata, sample_idx, ion_list)
        df = pd.concat([df,ion_peaks_df], axis = 0)
    
    # Join multi column index into one separated by -
    df.columns = df.columns.map(lambda x: '_'.join([str(i) for i in x]))
    
    return df


# ===== DEEP LEARNING =====
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
                               values='abun_minsub_scaled')
    
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
    
      

# === TARGET ENCODING ===


       






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