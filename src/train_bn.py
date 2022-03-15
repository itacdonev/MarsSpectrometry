#!/Users/itacdonev/opt/miniconda3/envs/nasamars

"""
Main training code
"""

# ===== ENVIRONMENT =====
from __future__ import print_function, absolute_import
import pandas as pd
from src import config
from termcolor import colored


def load_tables():
    metadata = pd.read_csv(config.DATA_DIR + 'metadata.csv')
    train_labels = pd.read_csv(config.DATA_DIR + 'train_labels.csv')
    submission = pd.read_csv(config.DATA_DIR + 'submission_format.csv')

    # Define samples
    train = metadata[metadata.split == 'train'].copy().reset_index(drop=True)
    valid = metadata[metadata.split == 'val'].copy().reset_index(drop=True)
    test = metadata[metadata.split == 'test'].copy().reset_index(drop=True)
    
    return metadata, train_labels, train, valid, test, submission



def main():
    
    # ===== DATA MANAGEMENT =====
    print(colored('Importing tables', 'blue'))
    metadata, train_labels, train, valid, test , submission = load_tables()
    print(f'TRAIN: {train.shape}')
    print(f'VALID: {valid.shape}')
    print(f'TEST: {test.shape}')
    print(f'SUBMISSION: {submission.shape}')
    
    # Get sample files
    train_files = train['features_path'].values
    valid_files = valid['features_path'].values
    test_files = test['features_path'].values

    # List of targets
    print(colored('\nTarget labels', 'blue'))
    targets = [col for col in train_labels if col not in ['sample_id']]
    print(f'Number of target labels: {len(targets)}')
    print(f'Targets: {targets}')
    
    
    
    
    
    
if __name__=="__main__":
    main()