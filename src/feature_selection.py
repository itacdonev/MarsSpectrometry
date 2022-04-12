"""Feature selection methods
"""

import os
import joblib, json
from tqdm import tqdm
from termcolor import colored
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import log_loss
from src import config, model_selection


class SelectModelFeatures():
    """
    Select features based on the trained model using
    SelectFromModel() method. Save new features as a dictionary
    for each label.
    
    Arguments
    ---------
    base_sfm_features_name (str): features file name of the input SFM features for training.
                                  Features from the base model.
    target_labels_list (list): list of target labels
    new_features_file_name (str): name of the new added feature data table
    fitted_model_name (str): name of the fitted model based on which
                            the threshold should be computed. The current model.
    fitted_model_algo (str): model algorithm on which the fitted_model_name
                            was trained.
    X_tr (pandas df): training data frame
    X_vlte (pandas df): validation data frame
    split_type (str): 'tr', 'trvl'
    train_labels: training labels data frame. Includes all target labels.
    valid_files (dict): dictionary of validation files
    valid_labels: validation labels data frame. Includes all target labels.
    
    """

    def __init__(self,
                 target_labels_list:list,
                 new_features_file_name:str,
                 fitted_model_name:str,
                 fitted_model_algo:str,
                 base_sfm_features_name:str,
                 X_tr,
                 X_vlte,
                 split_type:str,
                 train_labels,
                 valid_files,
                 valid_labels):

        self.target_labels_list = target_labels_list
        self.new_features_file_name = new_features_file_name
        self.fitted_model_name = fitted_model_name
        self.fitted_model_algo = fitted_model_algo
        self.base_sfm_features_name = base_sfm_features_name
        self.X_tr = X_tr
        self.X_vlte = X_vlte
        self.split_type = split_type
        self.train_labels = train_labels
        self.valid_files = valid_files
        self.valid_labels = valid_labels

    def save_features(self, file_name, new_features_dict):
        """
        Save features dictionary to a file.
        """
        with open(file_name, 'w') as file:
            file.write(json.dumps(new_features_dict))
        print(f'Saving {file_name}')


    def load_features(self):
        print(colored(f'Loading feature column names', 'blue'))
        path_cols = self.base_sfm_features_name + '_' +\
                    self.fitted_model_algo + '_' +\
                    self.split_type + '_SFM_COLS.txt'
        print(f'Reading {path_cols}')
        with open(path_cols) as json_file:
                new_features_dict = json.load(json_file)
        
        # Add features from add-on features: new_features_file_name
        if self.new_features_file_name:
            print(f'Adding features from {self.new_features_file_name}')
            # DF of new features
            dt = pd.read_csv(os.path.join(config.DATA_DIR_OUT,
                                            self.new_features_file_name + '_' +
                                            self.split_type + '.csv'))
            new_model_cols = dt.columns

            for col in new_model_cols:
                print(col)
                for label in self.target_labels_list:
                    print(label)
                    new_features_dict[label].append(col)
                    print(col in new_features_dict[label])

        return new_features_dict


    def optimal_threshold_fi(self):
        """
        Compute optimal threshold for feature importance.
        Using a fitted model, load the model and
        compute the optimal threshold using the model
        metric. For the trained model use the one trained
        only on TRAIN, to be able to validate on VALID.

        Returns
        -------
        Threshold value for feature importance for each label.
        Thresholds are saved with '_sfmt.csv' suffix.
        """
        print('Computing optimal threshold for each label')
        thrs_value = {}
        for label in tqdm(self.target_labels_list):
            #print(colored(f'LABEL: {label}', 'blue'))

            # Get the full model name
            MODEL_CLF = self.fitted_model_name + '_' + self.split_type + '_' + label + '.joblib.dat'

            # Load saved model and compute thresholds
            clf = joblib.load(os.path.join(config.CLF_DIR, MODEL_CLF))
            fi_min = clf.feature_importances_.min()
            fi_max = clf.feature_importances_.max()
            threshold = np.arange(fi_min, fi_max, 0.001)

            thrs_loss_best = 5
            thrs_best = None

            for thrs in threshold:
                selection = SelectFromModel(clf, threshold=thrs, prefit=True)
                selection_fts = selection.get_support()
                X_tr_sfm_cols = self.X_tr.columns[selection_fts]
                X_tr_sfm = self.X_tr[X_tr_sfm_cols].copy()
                #X_tr_sfm = selection.transform(self.X_tr)
                # train model
                clf_sfm = model_selection.models[self.fitted_model_algo]
                clf_sfm.fit(X_tr_sfm, self.train_labels[label])

                # Evaluate model
                select_X_test = selection.transform(self.X_vlte.iloc[:len(self.valid_files),:])
                y_pred = clf_sfm.predict_proba(select_X_test)[:,1]
                thrs_loss = log_loss(self.valid_labels[label], y_pred, labels=(0,1))
                #print(f'Threshold={thrs}, n={X_tr_sfm.shape[0]}, log-loss: {thrs_loss}')
                if thrs_loss < thrs_loss_best:
                    thrs_loss_best = thrs_loss
                    thrs_best = thrs
            thrs_value[label] = thrs_best

        df_thrs = pd.DataFrame.from_dict(thrs_value, orient='index')
        df_thrs.columns = [self.fitted_model_name]
        df_thrs.index = df_thrs.index.set_names('target')

        # Save the computed thresholds
        df_thrs.to_csv(os.path.join(config.MODELS_DIR, self.fitted_model_name + '_sfmt.csv'),
                                    index=True)

        return thrs_value


    def select_threshold_columns(self, thrs_value:dict):
        """Fit the models again on the computed threshold for each label
        and record the columns retained in the model for each label.
        
        Returns
        -------
        Returns a dictionary of selected features for each label.
        The dictionary is saved with extension '_SFM_COLS.txt'.
        Example : 'fts_mra_tempmz_sfm_tr_SFM_COLS.txt'
        """
        
        print('Refinting the model based on the threshold')
        sfm_loss = {}
        new_features_dict = {}
        for label in self.target_labels_list:
            print(colored(f'LABEL: {label}', 'blue'))

            # Load saved model
            MODEL_CLF = self.fitted_model_name + '_' + self.split_type + '_' + label + '.joblib.dat'
            clf = joblib.load(os.path.join(config.CLF_DIR, MODEL_CLF))
            
            selection = SelectFromModel(clf, threshold=thrs_value[label], prefit=True)
            X_tr_sfm = selection.transform(self.X_tr)
            # Save features for each label
            sfm_idx = selection.get_support(indices=True)
            new_features_dict[label] = [self.X_tr.columns[i] for i in sfm_idx]

            # train model
            clf_sfm = model_selection.models[self.fitted_model_algo]
            clf_sfm.fit(X_tr_sfm, self.train_labels[label])

            # Evaluate model
            select_X_test = selection.transform(self.X_vlte.iloc[:len(self.valid_files),:])
            y_pred = clf_sfm.predict_proba(select_X_test)[:,1]
            label_loss = log_loss(self.valid_labels[label], y_pred, labels=(0,1))
            #print(f'N={X_tr_sfm.shape[1]}, log-loss: {label_loss}')
            sfm_loss[label] = label_loss

        file_path = self.fitted_model_name + '_' + self.split_type + '_SFM_COLS.txt'
        self.save_features(file_path, new_features_dict)

        return sfm_loss, new_features_dict


    def remove_cols(self, fitted_model_features, cols_to_remove):
        """
        Remove features from an existing dictionary of features.

        Arguments
        ---------
        fitted_model_features (dict): Features names of the fitted base model for each label.
        cols_to_remove: Columns to remove due to larger log-loss.
        """
        # Loop through each target label and add feaure if there is any
        for label in self.target_labels_list:
            if label in cols_to_remove:
                #print(label)
                fts = cols_to_remove[label]
                fitted_model_features[label].remove(fts)

        return fitted_model_features


    def select_features(self, cv_new_model, compute_features:bool=True):
        """Select features of the newly trained model.

        Arguments
        ---------
        compute_features (bool; default=True): Should the feature selection be recomputed
                    or read in from the file. If there are chain models, then if any are
                    recomputed the thresholds should be recomputed.

        Returns
        -------
        sfm_columns (dict): dictionary of selected from model features

        """
        # Read in the features
        if not compute_features:
            path_cols = os.path.join(config.MODELS_DIR,
                             self.base_sfm_features_name + '_' +
                             self.split_type + '_SFM_COLS.txt')
            if os.path.exists(path_cols):
                print(f'Reading features from {path_cols}')
                with open(path_cols) as json_file:
                    sfm_columns = json.load(json_file)
        else:
            # Adding new features to existing
            if self.new_features_file_name:
                # DF of new features
                dt = pd.read_csv(os.path.join(config.DATA_DIR_OUT,
                                              self.new_features_file_name + '_' +
                                              self.split_type + '.csv'))
                new_model_cols = dt.columns
                print(f'New features from {self.new_features_file_name + "_" + self.split_type + ".csv"}')
                
                # Only 1 new feature was added - check CV loss difference
                if len(new_model_cols) == 1:
                    # Read in the CVloss of the base model
                    base_model_loss = pd.read_csv(os.path.join(config.MODELS_DIR,
                                                            self.fitted_model_name + '_' +
                                                            self.split_type + '_sfm_cvloss.csv'),
                                                index_col='target')
                    base_model_loss = base_model_loss.to_dict()[self.fitted_model_name + '_' +
                                                                self.split_type + '_sfm']
                    # Load fitted model features
                    path_cols = self.fitted_model_name + '_' + self.split_type + '_COLS.txt'
                    print(f'Reading fitted model {path_cols}')
                    
                    with open(path_cols) as json_file:
                            fitted_model_features = json.load(json_file)
                    
                    # Loop over each target and check logloss
                    # between models. Detect features to remove.
                    cols_to_remove = {}
                    for i in self.target_labels_list:
                        base_loss = base_model_loss[i]
                        new_loss = cv_new_model[i]
                        if new_loss > base_loss:
                            cols_to_remove[i] = new_model_cols[0]
                    #print(f'Features to remove: {cols_to_remove}')
                    sfm_columns = self.remove_cols(fitted_model_features, cols_to_remove)

                    file_path = self.fitted_model_name + '_' + self.split_type + '_SFM_COLS.txt'
                    self.save_features(file_path, sfm_columns)

                # More than 1 feature was added - recompute using optimal threshold
                else:
                    print(f'Recomputing current model features from fitted model (no fts > 1) {self.fitted_model_name}')
                    thrs_value = self.optimal_threshold_fi()
                    _, sfm_columns = self.select_threshold_columns(thrs_value)

            # Recomputing current model features
            else:
                print(f'Recomputing current model features from fitted model {self.fitted_model_name}')
                thrs_value = self.optimal_threshold_fi()
                _, sfm_columns = self.select_threshold_columns(thrs_value)

        return sfm_columns


    def show_no_fts_label(self, sfm_columns:dict):
        """Print number of features per label"""
        
        for label in self.target_labels_list:
            print(f'{label}: {len(sfm_columns[label])}')
    