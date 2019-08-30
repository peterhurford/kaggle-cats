import numpy as np
import pandas as pd

from datetime import datetime
from scipy.sparse import csr_matrix, hstack, vstack

from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as auc

from sklearn.linear_model import LogisticRegression
from target_encoding import TargetEncoderClassifier
import lightgbm as lgb


def print_step(step):
    print('[{}] {}'.format(datetime.now(), step))


def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model', n_folds=5):
    kf = KFold(n_splits=n_folds, random_state=42)
    fold_splits = kf.split(train)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0]))
    feature_importance_df = pd.DataFrame()
    i = 1
    for dev_index, val_index in fold_splits:
        print('Started ' + label + ' fold ' + str(i) + '/' + str(n_folds))
        if isinstance(train, pd.DataFrame):
            dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
        else:
            dev_X, val_X = train[dev_index], train[val_index]
        dev_y, val_y = target[dev_index], target[val_index]
        params2 = params.copy()
        pred_val_y, pred_test_y, importances = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            print(label + ' cv score {}: {}'.format(i, cv_score))
        if importances is not None and isinstance(train, pd.DataFrame):
            fold_importance_df = pd.DataFrame()
            fold_importance_df['feature'] = train.columns.values
            fold_importance_df['importance'] = importances
            fold_importance_df['fold'] = i
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        i += 1
    print('{} cv scores : {}'.format(label, cv_scores))
    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv total score : {}'.format(label, eval_fn(target, pred_train)))
    print('{} cv std score : {}'.format(label, np.std(cv_scores)))
    pred_full_test = pred_full_test / n_folds

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


def runFFLGB(train_X, train_y, test_X, test_y, test_X2, params):
    preds_test_y = []
    preds_test_y2 = []
    aucs = []
    if params.get('cat_cols'):
        cat_cols = params.pop('cat_cols')
    for col in train_X.columns:
        print(col)
        if col in cat_cols:
            print('(cat)')
            cat_col = [col]
        else:
            cat_col = []
        d_train = lgb.Dataset(train_X[[col]], label=train_y)
        d_valid = lgb.Dataset(test_X[[col]], label=test_y)
        watchlist = [d_train, d_valid]
        model = lgb.train(params,
                          train_set=d_train,
                          valid_sets=watchlist,
                          num_boost_round=1000,
                          early_stopping_rounds=40,
                          categorical_feature=cat_col,
                          verbose_eval=10)
        pred_test_y = model.predict(test_X[[col]], num_iteration=model.best_iteration)
        pred_test_y2 = model.predict(test_X2[[col]], num_iteration=model.best_iteration)
        aucs += [auc(test_y, pred_test_y)]
        preds_test_y += [pred_test_y]
        preds_test_y2 += [pred_test_y2]

    pred_test_y = np.mean(preds_test_y, axis=0)
    pred_test_y2 = np.mean(preds_test_y2, axis=0)
    import pdb
    pdb.set_trace()
    return pred_test_y, pred_test_y2, None


def runLR(train_X, train_y, test_X, test_y, test_X2, params):
    params['random_state'] = 42
    if params.get('scale'):
        print_step('Scale')
        params.pop('scale')
        scaler = StandardScaler()
        scaler.fit(train_X.values)
        train_X = scaler.transform(train_X.values)
        test_X = scaler.transform(test_X.values)
        test_X2 = scaler.transform(test_X2.values)

    print_step('Train LR')
    model = LogisticRegression(**params)
    model.fit(train_X, train_y)
    print_step('Predict 1/2')
    pred_test_y = model.predict_proba(test_X)[:, 1]
    print_step('Predict 2/2')
    pred_test_y2 = model.predict_proba(test_X2)[:, 1]
    return pred_test_y, pred_test_y2, None


def runTarget(train_X, train_y, test_X, test_y, test_X2, params):
    print_step('Train Target')
    model = TargetEncoderClassifier(**params)
    model.fit(train_X, train_y)
    print_step('Predict 1/2')
    pred_test_y = model.predict_proba(test_X)[:, 1]
    print_step('Predict 2/2')
    pred_test_y2 = model.predict_proba(test_X2)[:, 1]
    return pred_test_y, pred_test_y2, None


def runLRPL(train_X, train_y, test_X, test_y, test_X2, params):
    print_step('Train LR')
    model = LogisticRegression(**params)
    model.fit(train_X, train_y)
    print_step('Predict 1/2')
    pred_test_y = model.predict_proba(test_X)[:, 1]
    print_step('Predict 2/2')
    pred_test_y2 = model.predict_proba(test_X2)[:, 1]
    print('Before PL AUC: ' + str(auc(test_y, pred_test_y)))
    print_step('PL')
    test_small = test_X2[pred_test_y2 < 0.01]
    test_large = test_X2[pred_test_y2 > (1 - 0.01)]
    print('...PL Added {} rows'.format(test_small.shape[0] + test_large.shape[0]))
    train_X = vstack((train_X, test_small, test_large))
    train_y = pd.concat((train_y, pd.Series(0 for i in range(test_small.shape[0]))))
    train_y = pd.concat((train_y, pd.Series(1 for i in range(test_large.shape[0]))))
    train_y = train_y.values
    print_step('Train LRPL')
    model = LogisticRegression(**params)
    model.fit(train_X, train_y)
    print_step('Predict 1/2')
    pred_test_y = model.predict_proba(test_X)[:, 1]
    print_step('Predict 2/2')
    pred_test_y2 = model.predict_proba(test_X2)[:, 1]
    print('After PL AUC: ' + str(auc(test_y, pred_test_y)))
    return pred_test_y, pred_test_y2, None


def ohe(train, test, cat_cols, numeric_cols='auto'):
    print_step('Dummies 1/9')
    traintest = pd.concat([train, test])
    if isinstance(cat_cols, pd.Index) or isinstance(cat_cols, list) or len(set(traintest[cat_cols].values)) > 100:
        print_step('Dummies 2/9')
        if not isinstance(cat_cols, list) and not isinstance(cat_cols, pd.Index):
            cat_cols = [cat_cols]
        dummies = pd.get_dummies(traintest[cat_cols], columns=cat_cols, drop_first=True, sparse=True)
        print_step('Dummies 3/9')
        dummies = dummies.to_coo().tocsr()
        print_step('Dummies 4/9')
        print('Cats: {}'.format(sorted(cat_cols)))
        if numeric_cols == 'auto':
            numeric_cols = list(set(train.columns) - set(cat_cols))
        if numeric_cols:
            print('Numerics: {}'.format(sorted(numeric_cols)))
            numerics = traintest[numeric_cols]
            print_step('Dummies 5/9')
            scaler = StandardScaler()
            numerics = scaler.fit_transform(numerics.values)
            print_step('Dummies 6/9')
            numerics = csr_matrix(numerics)
            print_step('Dummies 7/9')
            dummies = hstack((numerics, dummies)).tocsr()
        else:
            print('...No numeric cols, skipping steps')
        print_step('Dummies 8/9')
        train_ohe = dummies[:train.shape[0], :]
        print_step('Dummies 9/9')
        test_ohe = dummies[train.shape[0]:, :]
    else:
        print_step('Dummies 2/9')
        dummies = pd.get_dummies(traintest[cat_cols], columns=cat_cols, drop_first=True)
        numeric_cols = list(set(train.columns) - set(cat_cols))
        if numeric_cols:
            numerics = traintest[numeric_cols]
            print_step('Dummies 3/9')
            scaler = StandardScaler()
            numerics = pd.DataFrame(scaler.fit_transform(numerics.values))
            print_step('Dummies 4/9')
            dummies = pd.concat([numerics.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        else:
            print('...No numeric cols, skipping steps')
        print_step('Dummies 8/9')
        train_ohe = dummies.iloc[:train.shape[0], :]
        print_step('Dummies 9/9')
        test_ohe = dummies.iloc[train.shape[0]:, :]
    return train_ohe, test_ohe
