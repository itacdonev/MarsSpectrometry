"""Feature engineering"""

import pandas as pd




# Compute average abundance for each sample
def compute_avg_abundance_ion(df_sample, sample_name:str):
    """ Compute average abundance per ion type"""
    dt = pd.DataFrame(df_sample.groupby('m/z')['abundance'].agg('mean')).reset_index()
    dt['sample_id'] = sample_name
    dt.columns = ['Ion_type', 'avg_abundance', 'sample_id']
    
    return dt


# Features are ions
def ion_avg_abundance():
    """Compute avergae abundance for each ion in the sample
    """
    
    # Initialize a table to store computed values
    dt = pd.DataFrame(dtype='float64')
    
    # Compute average abundance for each sample
    
    
    
    return dt