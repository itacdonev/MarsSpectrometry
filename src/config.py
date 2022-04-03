"""Global variables to be used for the project.
"""
import pandas as pd

# GLOBAL VARIABLES
DATA_DIR = '../input/'
DATA_DIR_OUT = '../output/'
MODELS_DIR = '../models/'
RANDOM_SEED = 42
NO_CV_FOLDS = 10
LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 10
PCA_COMPONENTS = None
CLASS_WEIGHTS = {0:0.01,
                 1:1.0}
INTENSITY_THRESHOLD = 0.01
MZ_THRESHOLD = 100