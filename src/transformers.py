from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm

class TransformWOE(BaseEstimator, TransformerMixin):
    """
    WOE (Weight of Evidence) transformation of variables.
    Transform abundance of each mz ratio.
    """
    
    def __init__(self, features, n_bins):
        self.features = features
        self.n_bins = n_bins
         
    def fit(self, X, y):
        
        self.woe_dict = []
        woe_transform = []
        
        for feature in self.features:
            #print(f'{feature}')
            #X[feature] = np.round(X[feature], decimals = 2)
            
            # Bin the series into quantiles given n_bins argument
            binned_f = pd.qcut(X[feature],
                            self.n_bins,
                            duplicates = 'drop',
                            precision = 0)
            
            X[feature + '_bin'] = binned_f
            #print(X)
            assert (X[feature + '_bin'] == binned_f).all()
            # Checks
            # dq = []
            # for n, value in enumerate(X[feature]):
            #     if value in X.loc[n, feature + '_bin']:
            #         dq.append(1)
            #     else: dq.append(0)
            # assert sum(dq) == X.shape[0], 'Some bins are not correctly assigned!'
            # del dq
            X.drop([c for c in X if c.endswith('_bin')], axis = 1, inplace = True)

            # Save data into a new df
            ht1 = pd.DataFrame({'x': binned_f,
                                'y': y})

            # Aggregate the table to get the count of examples and sum of bads for each range
            ht = ht1.groupby('x', as_index = False).agg({'y': ['count', 'sum']})
            del ht1
            
            # Rename columns
            ht.columns = ['Range', 'N', 'Events']
            
            # Compute all the values
            ht = ht.assign(Non_Events = lambda ht: ht['N'] - ht['Events'])\
                .assign(Perc_Bad = lambda ht: (ht['Events'] / ht['N'])*100)\
                .assign(Distr_N = lambda ht: ht['N'] / ht['N'].sum())\
                .assign(Distr_Bad = lambda ht: ht['Events'] / ht['Events'].sum())\
                .assign(Distr_Good = lambda ht: ht['Non_Events'] / ht['Non_Events'].sum())\
                .assign(WoE = lambda ht: np.log(ht['Distr_Good'] / ht['Distr_Bad']))\
                .assign(IV_range = lambda ht: (ht['Distr_Good'] - ht['Distr_Bad']) * ht['WoE'])\
                .assign(IV = lambda ht: ht['IV_range'].sum())\
                .round(3)
            
            # Create a dictionary of ranges and WoE values
            woe_dict_fts = dict(zip(ht['Range'], ht['WoE']))
            #---------------------------------------------
            # Replace the lower bound and upper bound with -np.Inf and np.Inf resp.
            # Get the number of woe bins
            n = len(list(woe_dict_fts.keys()))
            # Get 1st and last bin interval
            first_int = list(woe_dict_fts.keys())[0]
            last_int = list(woe_dict_fts.keys())[n-1]

            # Create new 1st and last bin interval
            # Delete the old intervals
            new1 = pd.Interval(left = -np.Inf, right = first_int.right, closed = 'right')
            woe_dict_fts[new1] = woe_dict_fts.pop(first_int)
            new5 = pd.Interval(left = last_int.left, right = np.Inf, closed = 'right')
            woe_dict_fts[new5] = woe_dict_fts.pop(last_int)
            
            # Order the final dictionary
            woe_dict_fts = collections.OrderedDict(sorted(woe_dict_fts.items()))
            #print('-'*45)
            #print(colored(f'{feature}', 'red'))
            #print(woe_dict_fts)
            #print('-'*45)
            #---------------------------------------------

            # Add woe_dict for the feature to the main table
            self.woe_dict.append(woe_dict_fts)

            del ht

            #print("-"*35)
            #print(feature)
            #print(woe_dict_fts)
            woe_transform.append((feature, woe_dict_fts))

        return self


    def transform(self, X, y = None):
        # Map WOE values

        for i, feature in enumerate(self.features):

            # Round the values to reduce the need for precision
            #X[feature] = np.round(X[feature], decimals = 2)

            # Map the WOE values
            X[feature] = X[feature].map(self.woe_dict[i])

            n = X[[feature]].drop_duplicates().reset_index(drop = True).shape[0]
            assert n <= self.n_bins, "Check precision in qcut binning!"

        return X
    