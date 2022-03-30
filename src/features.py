"""Feature engineering"""


from calendar import c
from importlib.metadata import metadata
from operator import index
import pandas as pd
import numpy as np
from src import config, preprocess, utils
from scipy.signal import find_peaks
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import spearmanr

from sklearn.metrics import auc
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import gc
from termcolor import colored


# BENCHMARK FEATURES
# Bin the temp and compute processes max abundance for each bin
def bin_temp_abund(df_sample, sample_name:str, detrend_method:str):
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
                                              detrend_method=detrend_method)
    
    # Compute max relative abundance
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


def features_iontemp_abun(df_meta, sample_list, detrend_method:str):
    
    # Initialize a table to store computed values
    dt = pd.DataFrame(dtype='float64')
    
    #ion_temp_dict = {}
    
    # Loop over all sample_id and compute. Add computation to dt.
    print(f'Number of samples: {len(sample_list)}')
    for i in sample_list:
        sample_name = df_meta.iloc[i]['sample_id']
        #print(sample_name)
        df_sample = preprocess.get_sample(df_meta, i)
        
        ht_pivot = bin_temp_abund(df_sample, sample_name, detrend_method)
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
     

def bin_temp_area(df_sample, sample_name, detrend_method:str):
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
    for ion in tqdm(ion_list):    
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
    
    
    # Prepare df so that each row is sample id
    df_pivot = pd.DataFrame.from_dict(ion_areas_dict)
    df_pivot.index = df_pivot.index.set_names('temp_bin')
    df_pivot = df_pivot.reset_index()
    df_pivot = df_pivot.melt('temp_bin')
    df_pivot['sample_id'] = sample_name
    df_pivot = df_pivot.pivot(index='sample_id', 
                              columns=['variable', 'temp_bin'], 
                              values='value')
    df_pivot.columns = df_pivot.columns.map(lambda x: '_'.join([str(i) for i in x]))
    df_pivot = df_pivot.add_prefix('Ion_')
    
    return df_pivot



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
    
    
    
def compute_ion_peaks(metadata, sample_idx, ion_list, detrend_method):
    
    # Select a sample and get sample name
    df_sample = preprocess.get_sample(metadata, sample_idx)
    sample_name = metadata.iloc[sample_idx]['sample_id']
    
    # Preprocess the sample
    df_sample = preprocess.preprocess_samples(df_sample, detrend_method=detrend_method)
    
    # Compute stats and save in dict for each ion type
    ion_peaks_cnt = {} # Initialize dictionary to save calculated values
    for ion in ion_list:
        #print(colored(f'ION: {ion}','blue'))
        ion_peaks_info = [] # initialize list to store stats per ion type
        
        temp_dt = df_sample[df_sample['m/z'] == ion].copy()
        
        # Apply Gaussian filter for the values
        temp_dt['abun_minsub_scaled_filtered'] = gaussian_filter1d(temp_dt['abun_scaled'], 
                                                                sigma=4)
        
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
def features_ion_peaks(file_paths:dict, metadata, ion_list:list, detrend_method):
    """
    Combines all computed ion peaks stats from each sample
    into a features data frame.
    """
    # Initialize a data frame to store all the sample calculations
    df = pd.DataFrame()
    
    for sample_idx in tqdm(file_paths):
        ion_peaks_df = compute_ion_peaks(metadata, sample_idx, ion_list, detrend_method)
        df = pd.concat([df,ion_peaks_df], axis = 0)
    
    # Join multi column index into one separated by -
    df.columns = df.columns.map(lambda x: '_'.join([str(i) for i in x]))
    
    return df

# ===== AREAS =====

def sample_abund_area(metadata, idx, detrend_method):
    # Compute the area under the abun_scaled for
    # the whole sample
    df_sample = preprocess.get_sample(metadata, idx)
    df_sample = preprocess.preprocess_samples(df_sample, detrend_method=detrend_method)
    df_sample = df_sample.sort_values(by=['time', 'abun_scaled'])
    
    x = df_sample['time'].values
    y = df_sample['abun_scaled'].values
    area = np.trapz(y=y,x=x)
    
    return area

def features_area(files, metadata, detrend_method):
    
    areas_dict = {}
    
    for idx in tqdm(files):
        sample_name = metadata.iloc[idx]['sample_id']
        area_sample = sample_abund_area(metadata, idx, detrend_method)
        areas_dict[idx] = area_sample
        
    return areas_dict
        

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
        