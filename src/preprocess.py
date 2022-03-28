"""Preprocess the data"""

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import minmax_scale
from src import config, preprocess
import pandas as pd
import numpy as np
from tqdm import tqdm


def get_sample(df_meta, i, verbose:bool=False):
    """Load one sample"""
    sample_file = df_meta.iloc[i]['features_path']
    sample_name = df_meta.iloc[i]['sample_id']
    if verbose:
        print(f'Sample ID: {sample_name}')
    df = pd.read_csv(config.DATA_DIR + sample_file)
    
    return df


def preprocess_ion_type(df):
    """
    Preprocess sample observations.
    """
    
    # Remove fractional values from the SAM testbed
    df = df[df['m/z'].transform(round) == df['m/z']]
    
    # Remove values of m/z greater than 99
    df = df[df['m/z'] < 100]
    
    # Remove all observations of the Helium carrier gas
    df = df[df['m/z'] != 4]
    
    return df


def remove_bcg_abund(df):
    """
    Subtracts minimum abundance value

    Args:
        df: dataframe with 'm/z' and 'abundance' columns

    Returns:
        dataframe with minimum abundance subtracted for all observations
    """

    df["abundance_minsub"] = df.groupby(["m/z"])["abundance"].transform(
        lambda x: (x - x.min())
    )
    del df['abundance']

    return df


def detrend_linreg(df_sample):
    """Detrending the background presence using linear regression."""
    X = df_sample['time'].values.reshape(-1,1)
    y = df_sample['abundance'].values
    
    model = LinearRegression()
    model.fit(X,y)
    
    trend = model.predict(X)
    
    df_sample['abundance_dtrend_lr'] = [y[i] - trend[i] for i in range(0,len(y))]
    # Replace negative values with zero - can't have neg mass
    df_sample['abundance_dtrend_lr'] = np.where(df_sample['abundance_dtrend_lr'] < 0, 
                                                0, 
                                                df_sample['abundance_dtrend_lr'])
    
    return df_sample
    


def scale_abun(df):
    """
    Scale abundance from 0-100 according to the min and max values across entire sample

    Args:
        df: dataframe containing abundances and m/z

    Returns:
        dataframe with additional column of scaled abundances
    """

    df["abun_minsub_scaled"] = minmax_scale(df["abundance_minsub"].astype(float))
    #TODO Shouldn't this be applied in the CV process???
    del df['abundance_minsub']
    
    return df


def preprocess_samples(df):
    # Preprocess m/z
    df = preprocess_ion_type(df)
    
    # Remove background abundance
    df = remove_bcg_abund(df)
    
    # MinMax scale abundance
    df = scale_abun(df)
    
    return df


def compute_min_max_temp_ion(metadata):
    """
    Compute min and max temperature for all samples.
    Compute unique values for the ion type.
    """
    min_temp = 0
    max_temp = 0
    ion_list = []
    
    for i in tqdm(range(metadata.shape[0])):
        # Load the sample
        df_sample = preprocess.get_sample(metadata, i)
        
        # Preprocess the sample data
        df_sample = preprocess_samples(df_sample)
        
        # Get the temp values
        if df_sample.temp.min() < min_temp:
            min_temp = df_sample.temp.min()
        if df_sample.temp.max() > max_temp:
            max_temp = df_sample.temp.max()
        
        # Get ion values
        sample_ions = df_sample['m/z'].unique()
        diff_ions = list(set(sample_ions).difference(set(ion_list)))
        if len(diff_ions) > 0:
            ion_list += diff_ions
        
    return min_temp, max_temp, ion_list

def compute_max_time_samples(metadata):
    """
    Compute maximum time across all samples in
    training, validation and test sets.
    """
    
    max_time = 0
    
    for i in tqdm(range(metadata.shape[0])):
        # Load the sample
        df_sample = preprocess.get_sample(metadata, i)
        df_sample = preprocess.preprocess_samples(df_sample)
        
        # Compute maximum time within the sample
        tm = df_sample.time.max()
        
        if tm > max_time:
            max_time = tm
            
    return max_time