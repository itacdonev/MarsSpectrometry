"""
Training functions including cross validation and full
model training.
"""
import os
from collections import Counter
import numpy as np
import pandas as pd
from termcolor import colored
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
import joblib
import lightgbm as lgb
import xgboost as xgb
from xgboost import plot_importance
from xgboost import plot_tree
import matplotlib.pyplot as plt
from src import config, features, model_selection

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
                  target_encode_fts:list=None,
                  fts_select:bool=None,
                  fts_select_cols:dict=None):
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
        print(colored(label, 'magenta'))
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
            #print(colored(f'FOLD {fold+1}', 'magenta'))
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

            # Estimate scale_pos_weight arg for XGB
            counter = Counter(y_train)
            estimate = counter[0] / counter[1]

            # ----- TARGET ENCODING -----
            if target_encode:
                #print('Encoding ...')
                if not target_encode_fts:
                    raise Exception(('Need to define which features to encode!'))
                else:
                    #print('Encoding ...')
                    temp = pd.concat([X_train.reset_index(drop=True), 
                                pd.Series(y_train, name=label)], 
                                axis=1)
                    # Loop over each feature to encode
                    for fts in target_encode_fts: 
                        #print(colored(fts, 'magenta'))
                        Xtr_te = features.label_encode(df=temp,
                                                        feature=fts,
                                                        target=label)
                        # Transform on train data
                        te_ft = Xtr_te.to_dict()
                        # Map encoding to train and valid
                        X_train[fts+'_te'] = X_train[fts].map(te_ft[label])
                        X_valid[fts+'_te'] = X_valid[fts].map(te_ft[label]).fillna(0)
                    
                # Delete original encoded columns
                Xtrain = X_train.copy()
                Xtrain.drop(target_encode_fts, axis=1, inplace=True)
                Xvalid = X_valid.copy()
                Xvalid.drop(target_encode_fts, axis=1, inplace=True)
            else:
                Xtrain = X_train.copy()
                Xvalid = X_valid.copy()

            # ----- Feature selection ----- 
            if fts_select:
                fts_columns = fts_select_cols[label]
                Xtrain = Xtrain[fts_columns].copy()
                Xtest = Xtest[fts_columns].copy()

            # Initialize the classifier
            if model_algo == 'XGB_imb':
                clf = model_selection.models[model_algo]
                clf.set_params(scale_pos_weight=estimate)
            else:
                clf = model_selection.models[model_algo]
            
            # Train a model
            clf.fit(Xtrain, y_train)

            # Compute predictions
            y_preds = clf.predict_proba(Xvalid)[:,1]

            # Compute model metric
            oof_logloss.append(model_metric(y_valid, y_preds))

        # Average log loss per label
        logloss[label] = np.sum(oof_logloss)/cv_folds

    if verbose:
        print(f'Average Log Loss: {np.mean(list(logloss.values()))}')
        print(logloss)

    return logloss


def train_full_model(X,
                     df_y,
                     target:list,
                     Xte,
                     sub_name,
                     model_algo:str,
                     target_encode:bool=None,
                     target_encode_fts:list=None,
                     test_sam:bool=False,
                     fts_select:bool=None,
                     fts_select_cols:dict=None
                     ):
    """
    Train full model
    
    model_algo: 'LR', 'XGB'
    """
    
    # Read in the submission file
    submission = pd.read_csv(config.DATA_DIR + 'submission_format.csv', 
                            index_col='sample_id')
    
    if test_sam:
        submission = submission.tail(12).copy()
    
    assert submission.shape[0] == Xte.shape[0]    
    
    # Get label names
    label_names = df_y[target]
    #clf_fitted_dict = {}              # fitted classifiers
    #df_te_fitted = pd.DataFrame()     # target encoding

    for label in label_names:

        # Select one label
        print(colored(f'LABEL: {label}', 'blue'))
        y = df_y[label].copy().values

        # estimate scale_pos_weight value for XGB
        counter = Counter(y)
        estimate = counter[0] / counter[1]

        # ----- TARGET ENCODING -----
        if target_encode:
            #print('Encoding ...')
            if not target_encode_fts:
                raise Exception(('Need to define which features to encode!'))
            else:
                temp = pd.concat([X.reset_index(drop=True), 
                                    pd.Series(y, name=label)], 
                                    axis=1)
                # Loop over each feature to encode
                for fts in target_encode_fts: 
                    #print(colored(fts, 'magenta'))
                    Xtr_te = features.label_encode(df=temp,
                                                    feature=fts,
                                                    target=label)
                    # Transform on train data
                    te_ft = Xtr_te.to_dict()
                    X[fts+'_te'] = X[fts].map(te_ft[label])
                    Xte[fts+'_te'] = Xte[fts].map(te_ft[label]).fillna(0)
            
            #print(X.head())
            #print(Xte.head())       
            
            # Delete original encoded columns
            Xtrain = X.copy()
            Xtrain.drop(target_encode_fts, axis=1, inplace=True)
            Xtest = Xte.copy()
            Xtest.drop(target_encode_fts, axis=1, inplace=True)
        else:
            Xtrain = X.copy()
            Xtest = Xte.copy()

        # ----- Feature selection ----- 
        if fts_select:
            fts_columns = fts_select_cols[label]
            Xtrain = Xtrain[fts_columns].copy()
            Xtest = Xtest[fts_columns].copy()
            
        # ----- MODEL SELECTION -----
        if model_algo == 'LR_reg':
            clf = LogisticRegression(penalty="l1",solver="liblinear", C=10, 
                                    random_state=config.RANDOM_SEED)
            
        elif model_algo == 'LGBM':
            clf = lgb.LGBMClassifier()
            
        elif model_algo == 'LGBM_opt':
            clf = lgb.LGBMClassifier(learning_rate=0.09,
                                     random_state=config.RANDOM_SEED)

        elif model_algo == 'XGB':
            clf = xgb.XGBClassifier(objective = "binary:logistic",
                                          use_label_encoder = False,
                                          eval_metric = 'logloss')
        elif model_algo == 'XGB_opt':
            clf = xgb.XGBClassifier(objective = "binary:logistic",
                                          use_label_encoder = False,
                                          eval_metric = 'logloss',
                                        learning_rate = 0.09)
        elif model_algo == 'XGB_imb':
            print(f'scale_pos_weight: {estimate}')
            clf = xgb.XGBClassifier(objective = "binary:logistic",
                                 use_label_encoder = False,
                                 eval_metric = 'logloss',
                                 learning_rate = 0.09,
                                 scale_pos_weight=estimate)
        elif model_algo == 'XGB_hp':
            clf = xgb.XGBClassifier(objective = "binary:logistic",
                                use_label_encoder = False,
                                eval_metric = 'logloss',                                
                                min_split_loss = 5)
        elif model_algo == 'SVC':
            clf = svm.SVC(probability=True)
            
        elif model_algo == 'PCA-XGB':
            clf = Pipeline([('PCA', PCA(n_components=config.PCA_COMPONENTS)),
                            ('XGB_opt', xgb.XGBClassifier(objective = "binary:logistic",
                                             use_label_encoder = False,
                                             eval_metric = 'logloss',
                                             learning_rate = 0.09))])
        elif model_algo == 'RFC':
            clf = RandomForestClassifier(random_state=config.RANDOM_SEED) 


        # ===== FIT THE MODEL FOR LABEL =====
        #print('Fit the model')
        #clf_fitted_dict[label] = clf.fit(Xtrain, y)
        clf.fit(Xtrain, y)
            
        # Feature importance plots
        if model_algo in ['XGB', 'XGB_opt', 'XGB_imb', 'XGB_hp', 'XGB_sfm']:
            _,ax = plt.subplots(1,1,figsize=(10,10))
            plot_importance(clf, max_num_features=30, height=0.5,
                            importance_type='gain', ax=ax)
            plt.show()
            #TODO install graphviz to plot tree
            #plot_tree(clf, num_trees=6)

        # save model to file
        joblib.dump(clf, os.path.join(config.MODELS_DIR,
                                      sub_name + '_' + label + ".joblib.dat"))

        # Make predictions
        submission[label] = clf.predict_proba(Xtest)[:,1]

    return submission


def train_tbl(df_train, df_labels,
              target_list,
              df_test,
              model_algo,
              sub_name:str,
              target_encode:bool=None,
              target_encode_fts:list=None,
              verbose:bool=False,
              test_sam:bool=False,
              fts_select:bool=None,
              fts_select_cols:dict=None):
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
    print(colored('CV training ....', 'blue'))
    train_cv_loss = trainCV_label(X=df_train,
                                  df_y=df_labels,
                                  target=target_list, 
                                  cv_folds=config.NO_CV_FOLDS,
                                  model_metric=log_loss,
                                  model_algo=model_algo,
                                  target_encode=target_encode,
                                  verbose=verbose,
                                  target_encode_fts=target_encode_fts,
                                  fts_select=fts_select,
                                  fts_select_cols=fts_select_cols)
    
    train_cv_loss_df = pd.DataFrame.from_dict(train_cv_loss, orient='index')
    train_cv_loss_df.columns = [sub_name]
    train_cv_loss_df.index = train_cv_loss_df.index.set_names('target')
    train_cv_loss_df.to_csv(os.path.join(config.MODELS_DIR, sub_name + '_cvloss.csv'),
                            index=True)
    
    # FULL TRAINING
    print(colored('Full training .....', 'blue'))
    submission = train_full_model(X=df_train,
                                  df_y=df_labels,
                                  Xte=df_test,
                                  sub_name=sub_name,
                                  target=target_list,
                                  model_algo=model_algo,
                                  target_encode=target_encode,
                                  target_encode_fts=target_encode_fts,
                                  test_sam=test_sam,
                                  fts_select=fts_select,
                                  fts_select_cols=fts_select_cols)
    
    # Save submission file
    submission.to_csv(config.MODELS_DIR + sub_name + '.csv')
    
    print(colored(f'\nAverage Log Loss: {np.round(np.mean(list(train_cv_loss.values())), 4)}', 'blue'))
    print('Log Loss per Label:')
    print(train_cv_loss)
    
    return train_cv_loss, submission


def compute_valid_loss(submission_file_VT,
                       valid_files,
                       valid_labels, 
                       target_label_list,
                       sub_name:str):
    """
    Compute validation loss.
    Model is trained only on TRAIN data set.
    Predictions are computed on the VALID data set.
    """
    df_sub = submission_file_VT.iloc[:len(valid_files),:]
    print(df_sub.shape)
    
    model_ll = {}
    for label in target_label_list:
        y_actual = valid_labels[label].iloc[:len(valid_files)]
        y_preds = df_sub[label].iloc[:len(valid_files)]
        model_ll[label] = log_loss(y_actual, y_preds, labels=(0,1))
        
    #print(f'Average Log Loss of full model: {np.mean(list(model_ll.values()))}')
    valid_loss = pd.DataFrame.from_dict(model_ll, orient='index')
    valid_loss.columns = [sub_name + 'V']
    valid_loss.index = valid_loss.index.set_names('target')
    valid_loss.to_csv(os.path.join(config.MODELS_DIR, sub_name + '_Vloss.csv'),
                            index=True)

    return model_ll, np.mean(list(model_ll.values()))
