"""Feature selection methods
"""

import os
import joblib, json
from tqdm import tqdm
from termcolor import colored
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import log_loss
from src import config, model_selection

def thrs_value_label(target_labels_list:list,
                     sfm_model_name:str,
                     sfm_feature:str,
                     model_algo:str,
                     X_tr,
                     X_vlte,
                     train_labels,
                     valid_files:dict,
                     valid_labels):
    """
    Using a fitted model, load the model and 
    compute the optimal threshold using the model
    metric. For the trained model use the one trained 
    only on TRAIN, to be able to validate on VALID.
    """

    thrs_value = {}
    for label in tqdm(target_labels_list):
        print(colored(f'LABEL: {label}', 'blue'))
        MODEL_CLF = sfm_model_name + '_' + label + '.joblib.dat'
        #VT_SAMPLE = sfm_feature + '_vlte.csv'
        # Load saved model and compute thresholds
        clf = joblib.load(os.path.join(config.MODELS_DIR, MODEL_CLF))
        fi_min = clf.feature_importances_.min()
        fi_max = clf.feature_importances_.max()
        threshold = np.arange(fi_min, fi_max, 0.001)
        #print(len(threshold))
        #print(threshold[:10])

        thrs_loss_best = 5
        thrs_best = None

        for thrs in threshold:        
            selection = SelectFromModel(clf, threshold=thrs, prefit=True)
            X_tr_sfm = selection.transform(X_tr)
            # train model
            clf_sfm = model_selection.models[model_algo]
            clf_sfm.fit(X_tr_sfm, train_labels[label])
            
            # Evaluate model
            select_X_test = selection.transform(X_vlte.iloc[:len(valid_files),:])
            y_pred = clf_sfm.predict_proba(select_X_test)[:,1]
            thrs_loss = log_loss(valid_labels[label], y_pred, labels=(0,1))
            print(f'Threshold={thrs}, n={X_tr_sfm.shape[0]}, log-loss: {thrs_loss}')
            if thrs_loss < thrs_loss_best:
                thrs_loss_best = thrs_loss
                thrs_best = thrs
        thrs_value[label] = thrs_best
    
    df_thrs = pd.DataFrame.from_dict(thrs_value, orient='index')
    df_thrs.columns = [sfm_model_name]
    df_thrs.index = df_thrs.index.set_names('target')
    df_thrs.to_csv(os.path.join(config.MODELS_DIR, sfm_model_name + '_sfmt.csv'),
                                index=True)
    
    return thrs_value


def save_features(fts_name, split_type, sfm_features_dict):
    """
    Save features dictionary to a file.
    """
    with open(fts_name + '_' + split_type + '_SFM_COLS.txt', 'w') as file:
        file.write(json.dumps(sfm_features_dict))    


def load_features(path_cols):
    with open(path_cols) as json_file:
            sfm_columns = json.load(json_file)
    return sfm_columns 
            
            
def fts_select_columns(target_labels_list:list,
                        sfm_model_name:str,
                        sfm_feature:str,
                        model_algo:str,
                        split_type:str,
                        X_tr,
                        X_vlte,
                        train_labels,
                        valid_files:dict,
                        valid_labels,
                        thrs_value:dict):
    sfm_loss = {}
    sfm_features_dict = {}
    for label in tqdm(target_labels_list):
        print(colored(f'LABEL: {label}', 'blue'))
        MODEL_CLF = sfm_model_name + '_' + split_type + '_' + label + '.joblib.dat'
        #VT_SAMPLE = sfm_feature + '_vlte.csv'
        # Load saved model and compute thresholds
        clf = joblib.load(os.path.join(config.MODELS_DIR, MODEL_CLF))
        
        selection = SelectFromModel(clf, threshold=thrs_value[label], prefit=True)
        X_tr_sfm = selection.transform(X_tr)
        # Save features for each label
        sfm_idx = selection.get_support(indices=True)
        sfm_features_dict[label] = [X_tr.columns[i] for i in sfm_idx]

        # train model
        clf_sfm = model_selection.models[model_algo]
        clf_sfm.fit(X_tr_sfm, train_labels[label])
        
        # Evaluate model
        select_X_test = selection.transform(X_vlte.iloc[:len(valid_files),:])
        y_pred = clf_sfm.predict_proba(select_X_test)[:,1]
        label_loss = log_loss(valid_labels[label], y_pred, labels=(0,1))
        print(f'N={X_tr_sfm.shape[1]}, log-loss: {label_loss}')
        sfm_loss[label] = label_loss
    
    save_features(sfm_feature, split_type, sfm_features_dict)
    
    # with open(sfm_feature + '_' + split_type + '_SFM_COLS.txt', 'w') as file:
    #     file.write(json.dumps(sfm_features_dict))
            
    return sfm_loss, sfm_features_dict

def fts_select(target_labels_list, sfm_feature, sfm_model_name, model_algo,
               split_type, X_tr, X_vlte, 
               train_labels, valid_files, valid_labels):
    # Check whether there exists a file with feature dictionary
    # for the base model. If so, read it in.
    path_cols = os.path.join(config.MODELS_DIR,
                             sfm_feature + '_' + split_type + '_SFM_COLS.txt')
    if os.path.exists(path_cols):
        print(f'Reading columns from {path_cols}')    
        with open(path_cols) as json_file:
            sfm_columns = json.load(json_file)
        return None, sfm_columns
    
    # If there is no file with features dict then compute
    # thresholds and then features
    else:
        # Check whether there are computed thresholds
        for i in ['_tr']:
            path = os.path.join(config.MODELS_DIR,
                                sfm_model_name + str(i) + '_sfmt.csv')
            print(path)
            if os.path.exists(path):
                print(f'Reading thresholds from {path}')
                df_temp = pd.read_csv(path, index_col='target')
                sfm_thrs = df_temp.to_dict()[sfm_model_name]
            else:
                print('Computing thresholds ...')
                sfm_thrs= thrs_value_label(
                            target_labels_list,
                            sfm_model_name + '_' + split_type,
                            sfm_feature,
                            model_algo,
                            X_tr,
                            X_vlte,
                            train_labels,
                            valid_files,
                            valid_labels)
    
    
        print('Computing columns ...')    
        sfm_loss, sfm_columns = fts_select_columns(
                        target_labels_list,
                        sfm_model_name,
                        sfm_feature,
                        model_algo,
                        split_type,
                        X_tr,
                        X_vlte,
                        train_labels,
                        valid_files,
                        valid_labels,
                        sfm_thrs)

        return sfm_loss, sfm_columns
    


def remove_cols(base_fts, cols_to_remove, target_labels_list):
    """
    Remove features from an existing dictionary of features. 
    """
    # Loop through each target label and add feaure if there is any
    for label in target_labels_list:
        if label in cols_to_remove:
            fts = cols_to_remove[label]
            base_fts[label].remove(fts)

    return base_fts


def update_fts_columns(target_labels_list,
                       sfm_feature,
                       sfm_model_name,
                        base_model,
                        model_algo,
                        model_suffix, 
                        cvloss, 
                        base_fts,
                        new_model_cols,
                        X_tr, X_vlte, 
                        train_labels, valid_files, valid_labels):
    """
    Check whether the cvloss is less than in the base
    model and if so add retain the feature. This is only
    valid for one feature data sets. For other use SelectFromModel().
    """
    
    if len(new_model_cols) == 1:    
        # Read in the CVloss of the base model
        base_model_loss = pd.read_csv(os.path.join(config.MODELS_DIR, base_model + '_' +
                                                model_suffix + '_cvloss.csv'), 
                                      index_col='target')
        base_model_loss = base_model_loss.to_dict()[base_model + '_' + model_suffix]

        # Loop over each target and check logloss
        # between models. Detect features to remove.
        cols_to_remove = {}
        for i in target_labels_list:
            base_loss = base_model_loss[i]
            new_loss = cvloss[i]
            if new_loss > base_loss:
                cols_to_remove[i] = new_model_cols[0]

        sfm_columns = remove_cols(base_fts, cols_to_remove, target_labels_list)
    
    else:
        # Run threshold computation and feature selection 
        # given a new model
        sfm_loss, sfm_columns = fts_select(target_labels_list, 
                                           sfm_feature, 
                                           sfm_model_name, 
                                           model_algo,
                                           model_suffix, X_tr, X_vlte, 
                                           train_labels, valid_files, valid_labels)
        
    return sfm_columns