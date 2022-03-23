from sklearn.model_selection import KFold,GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from termcolor import colored
from src import config, features, model_selection
import xgboost as xgb
import numpy as np
import pandas as pd
import os


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


def trainCV_label(X, df_y, 
                  target:list, 
                  cv_folds:str,
                  model_metric,
                  model_algo, 
                  verbose:bool=False,
                  target_encode:bool=False,
                  te_features:list=None):
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
    
    # TRAIN EACH LABEL SEPARATELY
    for label in label_names: 
        if verbose:
            print(colored(f'\nLABEL: {label}', 'blue'))
        
        # Select one label   
        y = df_y[label].copy()
        
        # Define cross validation
        cv = StratifiedKFold(n_splits = cv_folds,
                             random_state =config.RANDOM_SEED,
                             shuffle = True)
        
        # CROSS VALIDATION TRAINING
        oof_logloss = [] # Metric for each fold for one label
        
        # Define the folds and train the model
        for fold, (t_, v_) in enumerate(cv.split(X, y)):
            X_train = X.iloc[t_].copy()
            y_train = y.iloc[t_].values
            X_valid = X.iloc[v_].copy()
            y_valid = y.iloc[v_].values
            if verbose:
                print(colored(f'Training for FOLD = {fold + 1}', 'blue'))
                print(colored(f'X_train={X_train.shape}, y_train={y_train.shape}', 'magenta'))
                print(colored(f'X_valid={X_valid.shape}, y_valid={y_valid.shape}', 'magenta'))
                print(f'Event rate (TRAIN) = {np.round(y_train.sum()/y_train.shape[0],2)}%')
                print(f'Event rate (VALID) = {np.round(y_valid.sum()/y_valid.shape[0],2)}%')
            
            #le = LabelEncoder()
            #X_train['instrument_type'] = le.fit_transform(X_train['instrument_type'])
            #X_valid['instrument_type'] = le.transform(X_valid['instrument_type'])
    
            # Target encoding
            if target_encode:
                print('Encoding ...')
                temp = pd.concat([X_train.reset_index(drop=True), 
                                pd.Series(y_train, name=label)], 
                                axis=1)
                Xtr_te = features.label_encode(df=temp,
                                                feature='top_1',
                                                target=label)
                Xtr_te = Xtr_te.to_dict()
                X_train['top_1'] = X_train['top_1'].map(Xtr_te[label])
                X_valid['top_1'] = X_valid['top_1'].map(Xtr_te[label])
                
            # Traing the model
            clf = model_selection.models[model_algo]
            clf.fit(X_train, y_train)
            
            # Compute predictions
            y_preds = clf.predict_proba(X_valid)[:,1]
            
            # Compute model metric
            oof_logloss.append(model_metric(y_valid, y_preds))
        
        # Average log loss per label
        logloss[label] = np.sum(oof_logloss)/cv_folds

    if verbose:
        print(f'Average Log Loss: {np.mean(list(logloss.values()))}')
        print(logloss)
            
    return logloss


def train_full_model(X, df_y, target:list, model_algo:str):
    """
    Train full model
    
    model_algo: 'LR', 'XGB'
    """
    # Get label names
    label_names = df_y[target]
    clf_fitted_dict = {}
    
    for label in label_names: 
        
        # Select one label   
        #print(colored(f'LABEL: {label}', 'blue'))
        y = df_y[label].copy().values
        
        # Traing the model
        if model_algo == 'LR_reg':
            clf = LogisticRegression(penalty="l1",solver="liblinear", C=10, 
                                    random_state=config.RANDOM_SEED)

        elif model_algo == 'XGB':
            clf = clf = xgb.XGBClassifier(objective = "binary:logistic",
                                          use_label_encoder = False,
                                          eval_metric = 'logloss')
        elif model_algo == 'XGB_opt':
            clf = clf = xgb.XGBClassifier(objective = "binary:logistic",
                                          use_label_encoder = False,
                                          eval_metric = 'logloss',
                                        learning_rate = 0.09)
            
        clf_fitted_dict[label] = clf.fit(X, y)
        
    return clf_fitted_dict


def train_tbl(df_train, df_labels, target_list, df_test, 
              model_algo, sub_name:str, target_encode:bool=None,
              verbose:bool=False):
    """
    Train tabular data. The training is done on CV and full dataset.
    
    Arguments
    ---------
        df_train: pandas data frame or name of the saved table in
                  /output file folder.
    """
    # Read in the data
    if not isinstance(df_train, pd.DataFrame):
        df_train = pd.read_csv(os.path.join(config.DATA_DIR_OUT + 
                                            df_train + '.csv'))
    
    if not isinstance(df_test, pd.DataFrame):  
        df_test = pd.read_csv(os.path.join(config.DATA_DIR_OUT + 
                                            df_test + '.csv'))
        
    # CV TRAINING
    #print('CV training ....')
    train_cv_loss = trainCV_label(X=df_train,
                                  df_y=df_labels,
                                  target=target_list, 
                                  cv_folds=config.NO_CV_FOLDS,
                                  model_metric=log_loss,
                                  model_algo=model_algo,
                                  target_encode=target_encode,
                                  verbose=verbose)
    
    # FULL TRAINING
    #print('Full training .....')
    train_full_clf = train_full_model(X=df_train,
                                      df_y=df_labels,
                                      target=target_list,
                                      model_algo=model_algo)
    
    
    # SUBMISSION
    #print('Submission')
    submission = pd.read_csv(config.DATA_DIR + 'submission_format.csv', 
                             index_col='sample_id')
    for target in train_full_clf:
        clf = train_full_clf[target]
        submission[target] = clf.predict_proba(df_test)[:,1]
    
    submission.to_csv(config.MODELS_DIR + sub_name + '.csv')
    
    print(colored(f'\nAverage Log Loss: {np.round(np.mean(list(train_cv_loss.values())), 4)}', 'blue'))
    print('Log Loss per Label:')
    print(train_cv_loss)
    
    return train_cv_loss, train_full_clf, submission
