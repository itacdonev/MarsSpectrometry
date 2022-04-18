"""Global variables to be used for the project.
"""
# GLOBAL VARIABLES
DATA_DIR = '../input/'
DATA_DIR_OUT = '../output/'
MODELS_DIR = '../models/'
CLF_DIR = '../models/clf'
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
MZ_CNT_THRESHOLD = 100
CORRELATION_THRESHOLD = 0.6
NO_PEAKS_CALC = 4
FTS_GROUPS = ['fts_slope_tt','fts_mz_spectra', 'fts_mzstats',
              'fts_mass_loss_tempb', 'fts_cntpk_mratt',
              'fts_topmz', 'fts_peak_widths',
              'fts_mra_tempmz', 'fts_lr_corr_mz4',
               'fts_corr_mz4',
              'fts_mzstats', 'fts_mz_maxabun',
              'fts_range_abun_to_temp',]
WOE_BINS = 5