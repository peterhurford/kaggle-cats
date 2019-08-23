import numpy as np
import pandas as pd

from datetime import datetime
from scipy.sparse import csr_matrix, hstack

from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import KFold

import lightgbm as lgb


def print_step(step):
    print('[{}] {}'.format(datetime.now(), step))


def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model', classes=1):
    kf = KFold(n_splits=5)
    fold_splits = kf.split(train, target)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0], classes))
    feature_importance_df = pd.DataFrame()
    i = 1
    for dev_index, val_index in fold_splits:
        print('Started ' + label + ' fold ' + str(i) + '/5')
        if isinstance(train, pd.DataFrame):
            dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
            dev_y, val_y = target[dev_index], target[val_index]
        else:
            dev_X, val_X = train[dev_index], train[val_index]
            dev_y, val_y = target[dev_index], target[val_index]
        params2 = params.copy()
        pred_val_y, pred_test_y, importances = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        pred_val_y = pred_val_y.reshape(-1, classes)
        pred_test_y = pred_test_y.reshape(-1, classes)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            print(label + ' cv score {}: {}'.format(i, cv_score))
        if importances is not None:
            fold_importance_df = pd.DataFrame()
            fold_importance_df['feature'] = train.columns.values
            fold_importance_df['importance'] = importances
            fold_importance_df['fold'] = i
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        i += 1
    print('{} cv scores : {}'.format(label, cv_scores))
    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std score : {}'.format(label, np.std(cv_scores)))
    if classes == 1:
        pred_train = np.array([r[0] for r in pred_train])
        pred_full_test = np.array([r[0] for r in pred_full_test])
    pred_full_test = pred_full_test / 5.0

    results = {'label': label,
               'train': pred_train, 'test': pred_full_test,
               'cv': cv_scores,
               'importance': feature_importance_df}
    return results


def runLGB(train_X, train_y, test_X, test_y, test_X2, params):
    print_step('Prep LGB')
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    print_step('Train LGB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None
    if params.get('early_stop'):
        early_stop = params.pop('early_stop')
    if params.get('nbag'):
        nbag = params.pop('nbag')
    else:
        nbag = 1
    if params.get('cat_cols'):
        cat_cols = params.pop('cat_cols')
    else:
        cat_cols = []

    preds_test_y = []
    preds_test_y2 = []
    for b in range(nbag):
        params['seed'] += b
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop,
                          categorical_feature=cat_cols)
        print_step('Predict 1/2')
        pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
        print_step('Predict 2/2')
        pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
        preds_test_y += [pred_test_y]
        preds_test_y2 += [pred_test_y2]

    pred_test_y = np.mean(preds_test_y, axis=0)
    pred_test_y2 = np.mean(preds_test_y2, axis=0)
    return pred_test_y, pred_test_y2, model.feature_importance()
