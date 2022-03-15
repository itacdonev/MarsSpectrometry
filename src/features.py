"""Feature engineering"""

import pandas as pd
import numpy as np
from src import config, preprocess
from tqdm import tqdm



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