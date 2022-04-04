"""Preprocess the data"""

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import minmax_scale, PolynomialFeatures
from src import config, preprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter1d



def combine_samples(file_paths, metadata, 
                    detrend_method,
                    add_target:bool=False,
                    y_labels=None):
    """Combine sample into one data frame."""
    df = pd.DataFrame()
    for file in tqdm(file_paths):
        df_sample = preprocess.get_sample(metadata, file)
        df_sample = preprocess.preprocess_samples(df_sample, detrend_method=detrend_method)
        sample_name = metadata.iloc[file]['sample_id']
        df_sample['sample_id'] = sample_name
        df_sample['split'] = metadata.iloc[file]['split']
        if add_target:
            y = y_labels.iloc[file]['target']
            if len(y) > 0:
                df_sample['target'] = y[0]
            else: 
                df_sample['target'] = ''
        df = pd.concat([df, df_sample], axis=0)
    return df



def get_sample(df_meta, i, verbose:bool=False):
    """Load one sample"""
    sample_file = df_meta.iloc[i]['features_path']
    sample_name = df_meta.iloc[i]['sample_id']
    if verbose:
        print(f'Sample ID: {sample_name}')
    df = pd.read_csv(config.DATA_DIR + sample_file)
    
    return df


def preprocess_mz_value(df):
    """
    Preprocess sample observations.
    """
    
    # Remove fractional values from the SAM testbed
    df = df[df['m/z'].transform(round) == df['m/z']]
    
    # Remove values of m/z greater than 99
    df = df[df['m/z'] < config.MZ_THRESHOLD]
    
    # Remove all observations of the Helium carrier gas
    df = df[df['m/z'] != 4]
    
    return df


def remove_small_cnt_mz(df_sample, remove_mz_thrs):
    
    # Check how many in a sample are below the expected mean value of the train sample
    sample_mz_n_cnt = df_sample.groupby('m/z')['abundance'].agg('count')
    
    # which mz values are less than what is expected from the train sample
    small_cnt_mz = []
    for mz_idx in (sample_mz_n_cnt.index):
        s_mz = sample_mz_n_cnt[mz_idx]
        if s_mz < remove_mz_thrs:
            small_cnt_mz.append(mz_idx)

    # Remove insufficient m/z values
    if len(small_cnt_mz) > 0:
        #print(f'mz values to remove: {small_cnt_mz}')
        for mz in small_cnt_mz:
            #print(f'Removing {mz}')
            df_sample = df_sample[df_sample['m/z'] != mz]
            assert df_sample[df_sample['m/z'] == mz].empty
        
    return df_sample

         
    
def remove_bcg_abund(df):
    """
    Subtracts minimum abundance value

    Args:
        df: dataframe with 'm/z' and 'abundance' columns

    Returns:
        dataframe with minimum abundance subtracted for all observations
    """

    df["abundance_detrend"] = df.groupby(["m/z"])["abundance"].transform(
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
    
    df_sample['abundance_detrend'] = [y[i] - trend[i] for i in range(0,len(y))]
    # Replace negative values with zero - can't have neg mass
    df_sample['abundance_detrend'] = np.where(df_sample['abundance_detrend'] < 0, 
                                                0, 
                                                df_sample['abundance_detrend'])
    
    return df_sample
    

def detrend_poly(df_sample,n_degree:int=2):
    """Detrending the background presence using polynomial
    regression.
    """
    """Detrending the background presence using linear regression."""
    X = df_sample['time'].values.reshape(-1,1)
    y = df_sample['abundance'].values
    
    pf = PolynomialFeatures(degree=n_degree)
    Xp = pf.fit_transform(X)
    
    lr = LinearRegression()
    lr.fit(Xp,y)
    trend = lr.predict(Xp)
    
    df_sample['abundance_detrend'] = [y[i] - trend[i] for i in range(0,len(y))]
    df_sample['abundance_detrend'] = np.where(df_sample['abundance_detrend'] < 0, 
                                                0, 
                                                df_sample['abundance_detrend'])
    
    return df_sample


def scale_abun(df):
    """
    Scale abundance from 0-100 according to the min and max values across entire sample

    Args:
        df: dataframe containing abundances and m/z

    Returns:
        dataframe with additional column of scaled abundances
    """

    df["abun_scaled"] = minmax_scale(df["abundance_detrend"].astype(float))
    del df['abundance_detrend']
    
    return df


def smooth_mz_ts(df_sample, 
                 smoothing_type:str='gauss',
                 gauss_sigma:int=5,
                 ma_step:int=None):
    """
    Use Gaussian filter 1D to smooth the mz values.
    The sample should be preprocesses.
    
    smoothing_type (str, default='gauss', other possible values: 'ma')
    """
    
    if smoothing_type == 'gauss':
        df_sample['abun_scaled_smooth'] = df_sample.groupby('m/z')['abun_scaled']\
                                               .transform(lambda x: gaussian_filter1d(x, 
                                                                  sigma=gauss_sigma))
    else:
        df_sample['abun_scaled_smooth'] = df_sample.groupby('m/z')['abun_scaled']\
                                            .transform(lambda x: x.rolling(ma_step,1, center=True).mean())
    return df_sample


def preprocess_samples(df, 
                       detrend_method:str, 
                       poly_degree:int=2,
                       remove_mz_cnt:bool=False,
                       remove_mz_thrs=None,
                       smooth:bool=False,
                       smoothing_type:str='gauss',
                       gauss_sigma:int=5,
                       ma_step:int=None):
    # Preprocess m/z
    df = preprocess_mz_value(df)
    
    if remove_mz_cnt:
        #print(f'Removing mz ...')
        remove_small_cnt_mz(df, remove_mz_thrs=remove_mz_thrs)
    
    if detrend_method == 'min':
        # Remove background abundance
        df = remove_bcg_abund(df)
    elif detrend_method == 'lin_reg':
        df = detrend_linreg(df)
    elif detrend_method == 'poly':
        df =  detrend_poly(df, n_degree=poly_degree)
        
    # MinMax scale abundance
    df = scale_abun(df)
    
    # Smoothing
    if smooth:
        df = smooth_mz_ts(df, 
                          smoothing_type=smoothing_type,
                          gauss_sigma=gauss_sigma,
                          ma_step=ma_step)
        
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