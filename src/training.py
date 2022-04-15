"""
Training functions including cross validation and full
model training.
"""
from src import config, features, feature_selection, model_selection
import os, json, joblib
from collections import Counter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from termcolor import colored
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import svm
import lightgbm as lgb
import xgboost as xgb
from xgboost import plot_importance
from tqdm import tqdm
#from xgboost import plot_tree
import matplotlib.pyplot as plt

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


def trainCV_label(X,
                  df_y,
                  target:list,
                  cv_folds:str,
                  model_metric,
                  model_algo,
                  base_model_name:bool=False,
                  verbose:bool=False,
                  target_encode:bool=False,
                  target_encode_fts:list=None,
                  fts_select_cols:dict=None,
                  early_stopping:bool=False):
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

        base_model_name (str): Name of the base model.
                Example: 'fts_mra_tempmz_XGB_opt_tr'

        verbose: bool (default=False)
            If True it prints results for each fold.

    Returns
    -------

    """

    # Get label names
    label_names = df_y[target]

    # Get CV loss of the base model for comparison
    if base_model_name:
        file_path = os.path.join(config.MODELS_DIR, base_model_name + '_cvloss.csv')
        cv_base_model = pd.read_csv(file_path,
                                    index_col='target')
        cv_base_model = cv_base_model.to_dict()[cv_base_model.columns[0]]
        print(f'Basel model CVloss: {file_path}')
        
    # MODEL INFORMATION
    logloss = {}    # Average value of log loss for each label

    # TRAIN EACH LABEL SEPARATELY
    for label in label_names:

        if not base_model_name:
            print(colored(f'{label}', 'yellow'))

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
            if fts_select_cols:
                fts_columns = fts_select_cols[label]
                Xtrain = Xtrain[fts_columns].copy()
                Xvalid = Xvalid[fts_columns].copy()

            # Initialize the classifier
            if model_algo == 'XGB_imb':
                clf = model_selection.models[model_algo]
                clf.set_params(scale_pos_weight=estimate)
            else:
                clf = model_selection.models[model_algo]

            #print(f'Shape X: {Xtrain.shape[0]}')
            # Train a model
            if early_stopping:
                clf.fit(Xtrain, y_train,
                        early_stopping_rounds=10,
                        eval_metric="logloss",
                        eval_set=[(X_valid, y_valid)],
                        verbose=False)
            else:
                clf.fit(Xtrain, y_train)

            # Compute predictions
            y_preds = clf.predict_proba(Xvalid)[:,1]

            # Compute model metric
            oof_logloss.append(model_metric(y_valid, y_preds))

        # Average log loss per label
        logloss[label] = np.sum(oof_logloss)/cv_folds

        if base_model_name:
            ll_diff_to_base = np.round(logloss[label] - cv_base_model[label],5)
            if ll_diff_to_base < 0:
                print(colored(f'{label}: LogLoss={np.round(logloss[label],5)}', 'yellow'),
                    colored(f'-> {ll_diff_to_base}', 'green'))
            else:
                print(colored(f'{label}: LogLoss={np.round(logloss[label],5)}', 'yellow'),
                    colored(f'-> {ll_diff_to_base}', 'red'))
        else:
            print(colored(f'LogLoss {logloss[label]}', 'yellow'))

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
                     split_type:str,
                     target_encode:bool=None,
                     target_encode_fts:list=None,
                     test_sam:bool=False,
                     fts_select_cols:dict=None,
                     show_fi_plots:bool=False
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
    
    # To save the feature names for each label for the full model
    # without any selection
    feature_names_dict = {}
    for label in label_names:

        # Select one label
        #print(colored(f'{label}', 'yellow'))
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
        if fts_select_cols:
            fts_columns = fts_select_cols[label]
            Xtrain = Xtrain[fts_columns].copy()
            Xtest = Xtest[fts_columns].copy()
        
        # Add features
        feature_names_dict[label] = Xtrain.columns.tolist()
            
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
        print(colored(f'{label} - nfeatures: {len(Xtrain.columns)}', 'green'))
        clf.fit(Xtrain, y)
            
        # Feature importance plots
        if model_algo in ['XGB', 'XGB_opt', 'XGB_imb', 'XGB_hp', 'XGB_sfm']:
            if show_fi_plots:
                _,ax = plt.subplots(1,1,figsize=(10,5))
                plot_importance(clf, max_num_features=15, height=0.5,
                                importance_type='gain', ax=ax)
                plt.show()
                #TODO install graphviz to plot tree
                #plot_tree(clf, num_trees=6)

        # save model to file
        if fts_select_cols:
            joblib.dump(clf, os.path.join(config.CLF_DIR,
                                        sub_name + '_sfm_' + label + ".joblib.dat"))
        else:
            joblib.dump(clf, os.path.join(config.CLF_DIR,
                                        sub_name + '_' + label + ".joblib.dat"))
        # Make predictions
        submission[label] = clf.predict_proba(Xtest)[:,1]

    # Save feature names of the trained model
    if not fts_select_cols:
        file_name = sub_name + '_COLS.txt'
    else:
        file_name = sub_name + '_COLS_sfm.txt'
    with open(file_name, 'w') as file:
        file.write(json.dumps(feature_names_dict))
    print(f'Saving {file_name}')
    
    return submission


def train_tbl(df_train, df_labels,
              target_list,
              df_test,
              model_algo,
              sub_name:str,
              split_type:str,
              base_model_name:str=None,
              target_encode:bool=None,
              target_encode_fts:list=None,
              verbose:bool=False,
              test_sam:bool=False,
              fts_select_cols:dict=None,
              early_stopping:bool=False):
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
                                  base_model_name=base_model_name,
                                  target_encode=target_encode,
                                  verbose=verbose,
                                  target_encode_fts=target_encode_fts,
                                  fts_select_cols=fts_select_cols,
                                  early_stopping=early_stopping)
    
    train_cv_loss_df = pd.DataFrame.from_dict(train_cv_loss, orient='index')
    if fts_select_cols:
        train_cv_loss_df.columns = [sub_name + '_sfm']
    else:
        train_cv_loss_df.columns = [sub_name]
    train_cv_loss_df.index = train_cv_loss_df.index.set_names('target')
    if fts_select_cols:
        train_cv_loss_df.to_csv(os.path.join(config.MODELS_DIR, sub_name + '_sfm_cvloss.csv'),
                                index=True)
    else:
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
                                  split_type=split_type,
                                  target_encode=target_encode,
                                  target_encode_fts=target_encode_fts,
                                  test_sam=test_sam,
                                  fts_select_cols=fts_select_cols)

    # Save submission file
    if fts_select_cols:
        submission.to_csv(config.MODELS_DIR + sub_name + '_sfm.csv')
    else:
        submission.to_csv(config.MODELS_DIR + sub_name + '.csv')

    print(colored(f'CV LogLoss: {np.round(np.mean(list(train_cv_loss.values())), 5)}', 'yellow'))
    #print('Log Loss per Label:')
    #print(train_cv_loss)

    return train_cv_loss, submission


def compute_valid_loss(submission_file_VT,
                       valid_files,
                       valid_labels,
                       target_label_list,
                       sub_name:str,
                       fts_select_cols:str=None):
    """
    Compute validation loss.
    Model is trained only on TRAIN data set.
    Predictions are computed on the VALID data set.
    """
    df_sub = submission_file_VT.iloc[:len(valid_files),:]
    #print(df_sub.shape)
    
    model_ll = {}
    for label in target_label_list:
        y_actual = valid_labels[label].iloc[:len(valid_files)]
        y_preds = df_sub[label].iloc[:len(valid_files)]
        model_ll[label] = log_loss(y_actual.values, y_preds.values, labels=(0,1))

    #print(f'Average Log Loss of full model: {np.mean(list(model_ll.values()))}')
    valid_loss = pd.DataFrame.from_dict(model_ll, orient='index')
    if fts_select_cols:
        valid_loss.columns = [sub_name + '_sfm_V']
    else:
        valid_loss.columns = [sub_name + 'V']

    valid_loss.index = valid_loss.index.set_names('target')

    if fts_select_cols:
        valid_loss.to_csv(os.path.join(config.MODELS_DIR, sub_name + '_sfm_Vloss.csv'),
                                index=True)
    else:
        valid_loss.to_csv(os.path.join(config.MODELS_DIR, sub_name + '_Vloss.csv'),
                                index=True)

    return model_ll, np.mean(list(model_ll.values()))



# class TrainGroupSingle():
#     """
#     Train a single set of feature or feature by feature.
#     There are two different set of features: groups and individual.
#     If the feature is group type, then it should be trained and reduced
#     using one of the dimensinality reduction technique: SelectFromModel()
#     given an optimal threshold. 
#     If the feature is individual, then it should be tested against a base model 
#     individually and not as part of a group.
    
#     Arguments
#     ---------
    
#     """
    
#     def __init__(self):
#         pass

from sklearn.metrics import confusion_matrix
import seaborn as sns
def plot_conf_matrix(y_preds_class, y_actual_class, plot_cmat:bool=True):
    """
    Plot confusion matrix from the fitted model.
    """
    cf_matrix = confusion_matrix(y_actual_class, y_preds_class)
    
    if plot_cmat:
        ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
        ax.set_title('Confusion Matrix\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(['False','True'])
        ax.yaxis.set_ticklabels(['False','True'])

        ## Display the visualization of the Confusion Matrix.
        plt.show()
    
    return cf_matrix


def model_conf_mat(target_labels_list,
                   valid_labels,
                   model_submission):
    """Compute confusion matrix for the final
    model by adding all the individual confusion
    matrices.
    """
    model_cmat = pd.DataFrame()
    for label in target_labels_list:
        label_sub = model_submission.iloc[:valid_labels.shape[0]][[label]].copy()
        label_sub[label+'_c'] = np.where(label_sub[label] > 0.5,1,0)
        y_preds_class = label_sub[label+'_c'].values
        y_actual_class = valid_labels[label].values
        dt = pd.concat([pd.Series(y_preds_class), pd.Series(y_actual_class)], axis=1)
        dt.columns = ['prediction','actual']
        cfmat = pd.crosstab(dt.actual, dt.prediction)

        if not model_cmat.empty:
            model_cmat = cfmat
        else:
            model_cmat = model_cmat + cfmat

    return model_conf_mat

class TrainLabel():
    """
    Train each label separately feeding one feature group at a time.
    
    Arguments
    ---------
    
    """
    
    def __init__(self,
                 target_labels_list:list,
                 df_train,
                 df_train_labels,
                 df_valid,
                 feature_group:str,
                 split_type:str,
                 model_algo:str,
                 save_file_suffix:str):
        self.target_labels_list=target_labels_list
        self.df_train=df_train
        self.df_train_labels=df_train_labels
        self.df_valid=df_valid
        self.feature_group=feature_group
        self.split_type=split_type
        self.model_algo=model_algo
        self.save_file_suffix=save_file_suffix
    
    # ===== STEP 1. =====
    # Train against all feature groups to get the best base model
    def get_best_base_model_label(self):
        """
        Train each label with all feature groups and record
        the best model.
        
        Returns
        -------
        base_model_fts_group (dict) = Name of the feature group which is best in CVloss.
        
        """
        # Dict= label: best feature group name
        base_model_fts_group = {}

        for fts_name_group in self.feature_group: # E.g. fts_topmz
            
            # Name of the file to save submission
            model_name = fts_name_group + '_' + self.model_algo
            
            # Train all labels for the specified feature
            cvloss, submission_fts_name = train_tbl(
                df_train=self.df_train,
                df_labels=self.df_train_labels,
                target_list=self.target_labels_list,
                df_test=self.df_valid,
                split_type=self.split_type,
                model_algo=self.model_algo,
                sub_name=model_name + '_' + self.split_type
                )
            base_model_fts_group[fts_name_group] = cvloss
        return base_model_fts_group
        
    # Loop over features group list
    # select feature group
    # Acquire SFM cols
    # Add columns to the existing FTS_TRAIN_COLS
    # Train model - record CVL
    # If CVL < previous loss , 
    #       train full model record VLoss. 
    #       add features to MODEL FTS
        
    
def train_single_label(target_labels_list:list,
                        df_train,
                        df_valid,
                        feature_group:str,
                        model_algo:str,
                        split_type:str):
    """
    Train each label separately feeding one feature group at a time.
    
    Arguments
    ---------
    feature_group (str): List of feature names as they are saved in DATA_DIR_OUT
    
    """
    FTS_TESTED = []         # Feature groups which have been trained
    MODEL_FEATURES = []     # Feature groups for the final model
    
    # Train against all feature groups to get the best base model
    
    for FTS_GROUP in feature_group:
        # Read in the SFM columns for the new feature to train
        fts_group_file_name = FTS_GROUP + '_' + model_algo + '_' + split_type + '_SFM_COLS.csv'
        # Check whether there is a base model, i.e. whether we have trained any features
        # and retained them
        if len(MODEL_FEATURES) > 1:
            # We need to combine features
            pass