import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns


def annotate_countplot(sp, df: pd.DataFrame(), 
                       perc_height:float, font_size:int=10):
    """
    Add annotations to the seaborn countplot. Text annotations
    are added above each bar, centered. The height of each
    annotation is at the level of 2% of the length of the data.
    
    Parameters
    ----------
    sp: seaborn countplot
        Drawn countplot.
        
    df: pandas data frame
        Input data table
    
    font_size: int
        Font size. Default is 10.
        
    perc_height: float
        Percentage height that the text should be shown
        above the bar in the countplot.
    """
    
    for p in sp.patches:
        height = p.get_height()
    
        sp.text(p.get_x() + p.get_width()/2., 
                height + len(df) * perc_height, height,
                ha = 'center', fontsize = font_size)

         
def plot_mz_ts(df_sample,col_to_plot:str):
    
    ions = df_sample['m/z'].unique().tolist()

    for ion in ions:
        if df_sample[df_sample['m/z'] == ion]['abun_scaled'].max() > 0.02:
            plt.plot(df_sample[df_sample['m/z'] == ion]['temp'],
                    df_sample[df_sample['m/z'] == ion][col_to_plot], 
                    label=ion)            
        else:
            plt.plot(df_sample[df_sample['m/z'] == ion]['temp'],
                    df_sample[df_sample['m/z'] == ion][col_to_plot])            
    plt.legend()
    plt.xlabel('Temperature')
    plt.ylabel(col_to_plot)
    plt.title('Time series plot for each m/z ion')
    
    sns.despine()
    plt.show()
    
    
def plot_ms(df_sample, col_to_plot:str, target:str=None, sample_id:str=None):
    """Plot mass spectra."""
    
    # List of ions present in the sample
    ion_list = df_sample['m/z'].unique().tolist()
       
    for ion in ion_list:
        temp = df_sample[df_sample['m/z'] == ion]
        y = temp[col_to_plot].max()
        if y > 0.01:
            plt.plot([ion,ion], [0,y], label=ion)
            #plt.ylim(0,0.1)
        else:
            plt.plot([ion,ion], [0,y], c='#2D4B73')
            #plt.ylim(0,0.1)
     
    plt.legend()
    plt.title(f'Sample: {sample_id}   Target: {target}')
    plt.xlabel('m/z ion')
    sns.despine()


def plot_mic(X_tr, train_labels, label):
    mic = mutual_info_classif(X_tr, train_labels[label])
    
    t = pd.DataFrame()
    t = pd.concat([t, pd.Series(X_tr.columns), pd.Series(mic)], axis=1, ignore_index=True)
    t.columns = ['feature', 'mic']
    t = t.sort_values(by='mic', ascending=False).reset_index(drop=True)
    t = t.iloc[:30,:].copy()
    
    plt.subplots(1, figsize=(26, 1))
    sns.heatmap(np.array(t['mic'])[:,np.newaxis].T, cmap='Blues', cbar=False, linewidths=1, annot=True)
    plt.yticks([], [])
    plt.gca().set_xticklabels(t.feature, rotation=25, ha='right', fontsize=10)
    plt.suptitle(label, fontsize=18, y=1.2)
    plt.gcf().subplots_adjust(wspace=0.2)
    plt.show()