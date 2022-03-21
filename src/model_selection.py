"""
Model selection
"""

from sklearn.linear_model import LogisticRegression
from sklearn import svm
import xgboost as xgb
from src import config


models = {
    'LR_reg': LogisticRegression(penalty="l1",solver="liblinear", C=10, 
                                    random_state=config.RANDOM_SEED),
    
    'XGB': xgb.XGBClassifier(objective = "binary:logistic",
                             use_label_encoder = False,
                             eval_metric = 'logloss'),
    
    'XGB_opt': xgb.XGBClassifier(objective = "binary:logistic",
                                 use_label_encoder = False,
                                 eval_metric = 'logloss',
                                 learning_rate = 0.09),
    
    'SVC': svm.SVC(probability=True)
}