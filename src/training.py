from random import shuffle
from tabnanny import verbose
from sklearn.model_selection import KFold,GroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from termcolor import colored
from src import config

import numpy as np
import pandas as pd

def define_cvfolds(df, no_folds, SEED:int, 
                   group:str=None, strata:str=None):
    """
    Define cv folds and save in the dataframe.
    
    Parameters
    ----------
        df: pandas data frame
            Input datatable for training.
        
        group: str
            Variable which defines the groups. For example, ID column.
        
        SEED: int
            Random state.
        
        no_splits: int
            Number of folds.
            
    Returns
    -------
        df: pandas data frame
            Added column kfold with designated folds.

    """
    # Create the CV columns
    df['kfold'] = -1
    
    if group is not None:
        cv = GroupKFold(n_splits=no_folds)
    elif strata is not None:
        cv = StratifiedKFold(n_splits=no_folds, 
                             random_state=config.RANDOM_SEED, 
                             shuffle=True)
    else:
        cv = KFold(n_splits=no_folds, 
                   shuffle=True, 
                   random_state=SEED)
    
    X = df.drop(['target_single'], axis=1).copy()
    y = df['target_single']
    assert X.shape[0] == y.shape[0]
    
    for fold, (t_, v_) in enumerate(cv.split(X, y, groups=group)):
        df.loc[v_, 'kfold'] = fold
        
    return df



def trainCV(X, df_y, 
            target:list, 
            cv_folds:str,
            model_metric,
            clf, 
            verbose:bool=False):
    """
    Training pipeline 
        - tabular one target label at a time
        - SKF for each label target
    
    Parameters
    ----------
        X: pandas data frame
            Input datatable for training.
            
        df_y: pandas data frame
            Input labels for training
        
        features: list
            List of features to train on.
        
        target: list
            List of target columns
            
        cv_folds: int
            Number of CV folds to split the sample.            
        
        model_metric
            Define model metric to be used for training.
            
        clf
            Classifier for training
            
        verbose: bool (default=False)
            If True it prints results for each fold.
            
    Returns
    -------
        
            
    """
    
    # Get label names
    label_names = df_y[target]
    
    # MODEL INFORMATION
    logloss = {}    # Average value of log loss for each label
    label_predictions = pd.DataFrame()  # Average fold predictions for each label
    
    for label in label_names: 
        print(colored(f'\nLABEL: {label}', 'blue'))
        # Select one label   
        y = df_y[label].copy()
        
        # Define cross validation
        cv = StratifiedKFold(n_splits = cv_folds,
                             random_state =config.RANDOM_SEED,
                             shuffle = True)
        
        # CROSS VALIDATION TRAINING
        oof_logloss = [] # Metric for each fold for one label
        oof_predictions = pd.DataFrame()
        
        # Define the folds and train the model
        for fold, (t_, v_) in enumerate(cv.split(X, y)):
            #print(colored(f'Training for FOLD = {fold + 1}', 'blue'))
            X_train = X.iloc[t_].reset_index(drop=True)
            y_train = y.iloc[t_].values
            X_valid = X.iloc[v_].reset_index(drop=True)
            y_valid = y.iloc[v_].values
            if verbose:
                print(colored(f'X_train={X_train.shape}, y_train={y_train.shape}', 'magenta'))
                print(colored(f'X_valid={X_valid.shape}, y_valid={y_valid.shape}', 'magenta'))
                print(f'Event rate (TRAIN) = {np.round(y_train.sum()/y_train.shape[0],2)}%')
                print(f'Event rate (VALID) = {np.round(y_valid.sum()/y_valid.shape[0],2)}%')
            
            # Traing the model
            clf.fit(X_train, y_train)
            
            # Compute predictions
            y_preds = clf.predict_proba(X_valid)[:,1]
            if oof_predictions.empty:
                oof_predictions = pd.concat([oof_predictions, y_preds])
            else:
                oof_predictions = oof_predictions.add(y_preds)

            # Compute model metric
            oof_logloss.append(model_metric(y_valid, y_preds))
        
        # Average log loss per label
        logloss[label] = np.sum(oof_logloss)/cv_folds

        # Average predictions per label
        label_predictions[label] = oof_predictions/cv_folds
        
    return logloss, label_predictions