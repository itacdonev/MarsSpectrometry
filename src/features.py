"""Feature engineering"""

import pandas as pd
from src import config, preprocess


# Compute average abundance for each sample
def compute_avg_abundance_ion(df_sample, sample_name:str):
    """ Compute average abundance per ion type"""
    dt = pd.DataFrame(df_sample.groupby('m/z')['abundance'].agg('mean')).reset_index()
    dt['sample_id'] = sample_name
    dt.columns = ['Ion_type', 'avg_abundance', 'sample_id']
    
    return dt


# Features are ions
def ion_avg_abundance(df_meta):
    """Compute avergae abundance for each ion in the sample
    """
    
    # Initialize a table to store computed values
    dt = pd.DataFrame(dtype='float64')
    
    
    # Loop over all sample_id and compute. Add computation to dt.
    sample_list = df_meta[df_meta.split == 'train']['sample_id'].to_list()
    print(f'Number of samples: {len(sample_list)}')
    
    for i, sample in enumerate(sample_list):
        
        df_sample = preprocess.get_sample(df_meta, i)
        
        # Preprocess data sample
        df_sample = preprocess.preprocess_samples(df_sample)
        
        # Compute average abundance for each sample
        dt_avg_abund = compute_avg_abundance_ion(df_sample, sample_name=sample)
        
        # Add computed values back to original data frame
        dt = pd.concat([dt, dt_avg_abund], ignore_index=True)
        
    return dt


def avg_temp_sample(df, df_files):
    # Average temperature per sample
    avg_temp_sample = {}
    for i in df_files:
        df = pd.read_csv(config.DATA_DIR + df_files[i])
        avg_temp_sample[i] = df.temp.mean()
        
    df['avg_temp'] = df.index.map(avg_temp_sample)
    
    return df