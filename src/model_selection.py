"""
Model selection
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb
from src import config


models = {
    'LR_reg': LogisticRegression(penalty="l1",solver="liblinear", C=10, 
                                    random_state=config.RANDOM_SEED),
    
    'LR_imb': LogisticRegression(penalty="l1",
                                 solver="liblinear", 
                                 C=10, 
                                 random_state=config.RANDOM_SEED,
                                 class_weight=config.CLASS_WEIGHTS),
    
    'RFC': RandomForestClassifier(random_state=config.RANDOM_SEED),
    
    'LGBM': lgb.LGBMClassifier(),
    
    'LGBM_opt': lgb.LGBMClassifier(learning_rate=0.09,
                                   random_state=config.RANDOM_SEED),
    
    'XGB': xgb.XGBClassifier(objective = "binary:logistic",
                             use_label_encoder = False,
                             eval_metric = 'logloss'),
    
    'XGB_imb': xgb.XGBClassifier(objective = "binary:logistic",
                                 use_label_encoder = False,
                                 eval_metric = 'logloss',
                                 learning_rate = 0.09),
    
    'XGB_opt': xgb.XGBClassifier(objective = "binary:logistic",
                                 use_label_encoder = False,
                                 eval_metric = 'logloss',
                                 learning_rate = 0.09),
        
    'XGB_hp': xgb.XGBClassifier(objective = "binary:logistic",
                                use_label_encoder = False,
                                eval_metric = 'logloss',
                                min_split_loss = 5),
    
    'SVC': svm.SVC(probability=True),
    'PCA-XGB': Pipeline([('PCA', PCA(n_components=config.PCA_COMPONENTS)),
                        ('XGB_opt', xgb.XGBClassifier(objective = "binary:logistic",
                                            use_label_encoder = False,
                                            eval_metric = 'logloss',
                                            learning_rate = 0.09))])
}
