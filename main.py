#!/Users/itacdonev/opt/miniconda3/envs/nasamars

"""
Main training code
"""

# ===== ENVIRONMENT =====
import pandas as pd
from src import config


def main():
    # ===== DATA MANAGEMENT =====
    # IMPORT DATA
    metadata = pd.read_csv(config.DATA_DIR + 'metadata.csv')
    train_labels = pd.read_csv(config.DATA_DIR + 'train_labels.csv')
    submission = pd.read_csv(config.DATA_DIR + 'submission_format.csv')

    # Dictionary of all train files and their sample id
    train_files = metadata[metadata.split == 'train']['features_path'].to_dict()
    valid_files = metadata[metadata.split == 'val']['features_path'].to_dict()
    test_files = metadata[metadata.split == 'test']['features_path'].to_dict()

    # Define samples
    train = metadata[metadata.split == 'train'].copy().reset_index(drop=True)
    valid = metadata[metadata.split == 'valid'].copy().reset_index(drop=True)
    test = metadata[metadata.split == 'test'].copy().reset_index(drop=True)

    
    
    
if __name__=="__main__":
    main()