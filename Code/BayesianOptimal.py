# BayesianOptimal.py
"""
Author: Zhang Lu
Version: 1.0.0
Date:2025-02-14
Description: Happy Valentine's Day.
"""

from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from skopt.space import Categorical
import numpy as np
import json

def run_bayesian_optimization(train_x_dist, train_y, test_x_dist, test_y, vldt_x_dist, vldt_y):
    result_df = pd.DataFrame(columns=['num_leaves', 'n_estimators', 'max_depth', 'learning_rate', 'subsample',
                                      'colsample_bytree', 'boosting_type', 'reg_alpha', 'reg_lambda', 'max_bin',
                                      'mean_score'])

    def rf_cv_lgb(num_leaves, n_estimators, max_depth, learning_rate, subsample,
                  colsample_bytree, boosting_type, reg_alpha, reg_lambda, max_bin):
        model = LGBMClassifier(num_leaves=int(num_leaves),
                               n_estimators=int(n_estimators),
                               max_depth=int(max_depth),
                               learning_rate=learning_rate,
                               subsample=subsample,
                               colsample_bytree=colsample_bytree,
                               boosting_type=boosting_type,
                               reg_alpha=reg_alpha,
                               reg_lambda=reg_lambda,
                               max_bin=int(max_bin),
                               n_jobs=1,
                               random_state=42,
                               verbose=-1)
        train_scores = cross_val_score(model, train_x_dist, train_y, cv=8, scoring='f1')
        model.fit(train_x_dist, train_y)
        test_scores = model.score(test_x_dist, test_y)
        oot_scores = model.score(vldt_x_dist, vldt_y)

        ######### modify target object accordingly #########
        # mean_score = np.mean(train_scores)
        mean_score = np.mean(train_scores)*0.5 + np.mean(test_scores)*0.25 + np.mean(oot_scores)*0.25
        # mean_score = np.std([np.mean(train_scores), np.mean(test_scores), np.mean(oot_scores)])
        ########## end #########

        result_df.loc[len(result_df)] = [num_leaves, n_estimators, max_depth, learning_rate, subsample,
                                         colsample_bytree, boosting_type, reg_alpha, reg_lambda, max_bin, mean_score]
        return mean_score

    bayes_lgb = BayesianOptimization(rf_cv_lgb, {'num_leaves': (4, 80),
                                                 'n_estimators': (50, 150),
                                                 'max_depth': (3, 6),
                                                 'learning_rate': (0.05, 0.3),
                                                 'subsample': (0.6, 0.9),
                                                 'colsample_bytree': (0.6, 0.9),
                                                 'boosting_type': Categorical(['goss', 'gbdt']),
                                                 'reg_alpha': (0.5, 10.0),
                                                 'reg_lambda': (0.5, 10.0),
                                                 'max_bin': (50, 200)},
                                 random_state=42)
    # bayes_lgb.set_bounds(new_bounds={'boosting_type': Categorical(['goss', 'gbdt'])}) # 分类变量的第二种写法
    bayes_lgb.maximize(init_points=5, n_iter=15)

    print(result_df)
    print(bayes_lgb.max)

    best_params = bayes_lgb.max['params']
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['max_bin'] = int(best_params['max_bin'])

    with open('json/best_params.json', 'w') as f:
        json.dump(best_params, f)

    # Create a new model instance with the best parameters
    model = LGBMClassifier(**best_params,
                           n_jobs=1,
                           random_state=42,
                           verbose=-1)
    return model, result_df, best_params