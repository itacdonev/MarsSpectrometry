"""Preprocess the data"""

from sklearn.preprocessing import minmax_scale
from src import config, preprocess
import pandas as pd
from tqdm import tqdm


def get_sample(df_meta, i, verbose:bool=False):
    """Load one sample."""
    sample_file = df_meta.iloc[i]['features_path']
    sample_name = df_meta.iloc[i]['sample_id']
    if verbose:
        print(f'Sample ID: {sample_name}')
    df = pd.read_csv(config.DATA_DIR + sample_file)
    
    return df


def compute_min_max_temp(metadata):
    """
    Compute min and max temperature for all samples
    """
    min_temp = 0
    max_temp = 0

    for i in tqdm(range(metadata.shape[0])):
        df_sample = preprocess.get_sample(metadata, i)
        if df_sample.temp.min() < min_temp:
            min_temp = df_sample.temp.min()
        if df_sample.temp.max() > max_temp:
            max_temp = df_sample.temp.max()
        
    print(f'Min temp = {min_temp}; Max temp = {max_temp}')
    return min_temp, max_temp

    
def preprocess_ion_type(df):
    """
    Preprocess sample observations.
    """
    
    # Fractional values from the SAM testbed
    df = df[df['m/z'].transform(round) == df['m/z']]
    
    # Remove values of m/z greater than 99
    df = df[df['m/z'] < 100]
    
    # Remove all observations of the Helium carrier gas
    df = df[df['m/z'] != 4]
    
    return df


def remove_background_abundance(df):
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

    return df


def scale_abun(df):
    """
    Scale abundance from 0-100 according to the min and max values across entire sample

    Args:
        df: dataframe containing abundances and m/z

    Returns:
        dataframe with additional column of scaled abundances
    """

    df["abun_minsub_scaled"] = minmax_scale(df["abundance_minsub"].astype(float))

    return df


def preprocess_samples(df):
    # Preprocess m/z
    df = preprocess_ion_type(df)
    
    # Remove background abundance
    df = remove_background_abundance(df)
    
    # MinMax scale abundance
    df = scale_abun(df)
    
    return df