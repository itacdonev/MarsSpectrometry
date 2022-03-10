import pandas as pd
import matplotlib.pyplot as plt
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
        
