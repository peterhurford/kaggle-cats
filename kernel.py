STOP_AT_DATASET = False
RUN_LR_W_LABEL = False
RUN_LGB_W_FREQ = False
RUN_LGB_W_LABEL = False
RUN_LGB_W_LGB = False
RUN_LGB_WITH_LR_ENCODING = False
RUN_LR_WITH_OHE = False
RUN_LR_WITH_ALL_OHE = False
RUN_LR_WITH_ALL_OHE_PLUS_SCALARS = True
RUN_LR_WITH_ALL_OHE_PLUS_SCALARS_20_FOLD = False
RUN_LR_WITH_ALL_OHE_PLUS_SCALARS_100_FOLD = False

ADD_LR = False
PRINT_LGB_FEATURE_IMPORTANCE = False


import string

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score as auc

from utils import print_step, ohe, run_cv_model, runLGB, runLR


print_step('Loading')
train = pd.read_csv('train.csv')
print(train.shape)
test = pd.read_csv('test.csv')
print(test.shape)


print_step('Subsetting')
target = train['target']
train_id = train['id']
test_id = test['id']
train.drop(['target', 'id'], axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)


print_step('Cleaning ordinal')
ord_1 = ['Contributor', 'Novice', 'Expert', 'Master', 'Grandmaster']
ord_1 = dict(zip(ord_1, range(len(ord_1))))
train.loc[:, 'ord_1'] = train['ord_1'].apply(lambda x: ord_1[x]).astype(int)
test.loc[:, 'ord_1'] = test['ord_1'].apply(lambda x: ord_1[x]).astype(int)

ord_2 = ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']
ord_2 = dict(zip(ord_2, range(len(ord_2))))
train.loc[:, 'ord_2'] = train['ord_2'].apply(lambda x: ord_2[x]).astype(int)
test.loc[:, 'ord_2'] = test['ord_2'].apply(lambda x: ord_2[x]).astype(int)

ord_3 = sorted(list(set(train['ord_3'].values)))
ord_3 = dict(zip(ord_3, range(len(ord_3))))
train.loc[:, 'ord_3'] = train['ord_3'].apply(lambda x: ord_3[x]).astype(int)
test.loc[:, 'ord_3'] = test['ord_3'].apply(lambda x: ord_3[x]).astype(int)

ord_4 = sorted(list(set(train['ord_4'].values)))
ord_4 = dict(zip(ord_4, range(len(ord_4))))
train.loc[:, 'ord_4'] = train['ord_4'].apply(lambda x: ord_4[x]).astype(int)
test.loc[:, 'ord_4'] = test['ord_4'].apply(lambda x: ord_4[x]).astype(int)

ord_5 = sorted(list(set(train['ord_5'].values)))
ord_5 = dict(zip(ord_5, range(len(ord_5))))
train.loc[:, 'ord_5'] = train['ord_5'].apply(lambda x: ord_5[x]).astype(int)
test.loc[:, 'ord_5'] = test['ord_5'].apply(lambda x: ord_5[x]).astype(int)


if STOP_AT_DATASET:
    import pdb
    pdb.set_trace()


print_step('Simple label encoding')
cat_cols = [c for c in train.columns if 'nom' in c]
for c in cat_cols + ['bin_3', 'bin_4']:
    le = LabelEncoder()
    le.fit(pd.concat([train[c], test[c]])) 
    train[c] = le.transform(train[c])
    test[c] = le.transform(test[c])


lr_params = {'solver': 'lbfgs', 'C': 0.1151, 'max_iter': 1000}
if RUN_LR_W_LABEL:
    lr_params2 = lr_params.copy()
    lr_params2['scale'] = True
    results = run_cv_model(train, test, target, runLR, lr_params2, auc, 'lr-label')


if RUN_LR_WITH_ALL_OHE or RUN_LR_WITH_ALL_OHE_PLUS_SCALARS or RUN_LR_WITH_ALL_OHE_PLUS_SCALARS_20_FOLD or RUN_LR_WITH_ALL_OHE_PLUS_SCALARS_100_FOLD:
    print_step('All OHE')
    train_ohe, test_ohe = ohe(train, test, train.columns)
    print(train_ohe.shape)
    print(test_ohe.shape)


if RUN_LR_WITH_ALL_OHE:
    results_lr = run_cv_model(train_ohe, test_ohe, target, runLR, lr_params, auc, 'lr-all-ohe')


if RUN_LR_WITH_ALL_OHE_PLUS_SCALARS or RUN_LR_WITH_ALL_OHE_PLUS_SCALARS_20_FOLD or RUN_LR_WITH_ALL_OHE_PLUS_SCALARS_100_FOLD:
    numeric_cols = list(set(train.columns) - set(cat_cols))
    traintest = pd.concat([train, test])
    numerics = traintest[numeric_cols]
    scaler = StandardScaler()
    numerics = scaler.fit_transform(numerics.values)
    numerics = csr_matrix(numerics)
    train_numerics = numerics[:train.shape[0], :]
    test_numerics = numerics[train.shape[0]:, :]
    train_ohe = hstack((train_numerics, train_ohe)).tocsr()
    test_ohe = hstack((test_numerics, test_ohe)).tocsr()
    print(train_ohe.shape)
    print(test_ohe.shape)
    if RUN_LR_WITH_ALL_OHE_PLUS_SCALARS_100_FOLD:
        n_folds = 100
    elif RUN_LR_WITH_ALL_OHE_PLUS_SCALARS_20_FOLD:
        n_folds = 20
    else:
        n_folds = 5
    results_lr = run_cv_model(train_ohe, test_ohe, target, runLR, lr_params, auc, 'lr-all-ohe', n_folds=n_folds)


if RUN_LR_WITH_OHE:
    print_step('OHE')
    train_ohe, test_ohe = ohe(train, test, cat_cols)
    print(train_ohe.shape)
    print(test_ohe.shape)
    results_lr = run_cv_model(train_ohe, test_ohe, target, runLR, lr_params, auc, 'lr-ohe')


lgb_params = {'application': 'binary',
              'boosting': 'gbdt',
              'metric': 'auc',
              'num_leaves': 24,
              'max_depth': 11,
              'learning_rate': 0.02,
              'bagging_fraction': 0.9,
              'feature_fraction': 0.3,
              'min_split_gain': 0.02,
              'min_child_samples': 150,
              'min_child_weight': 0.02,
              'verbosity': -1,
              'seed': 1,
              'lambda_l1': 0.1,
              'lambda_l2': 0.1,
              'early_stop': 100,
              'verbose_eval': 50,
              'num_rounds': 100000}


if ADD_LR:
    train.loc[:, 'lr'] = results_lr['train']
    test.loc[:, 'lr'] = results_lr['test']
    lgb_params['num_leaves'] = 4


if RUN_LGB_W_LABEL:
    results = run_cv_model(train, test, target, runLGB, lgb_params, auc, 'lgb-label')


if RUN_LGB_W_LGB:
    for col in cat_cols:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
    lgb_params2 = lgb_params.copy()
    lgb_params2['max_cat_to_onehot'] = 2
    lgb_params2['cat_smooth'] = 20
    lgb_params2['num_leaves'] = 2
    lgb_params2['learning_rate'] = 0.3
    lgb_params2['lambda_l1'] = 2.0
    lgb_params2['lambda_l2'] = 2.0
    lgb_params2['feature_fraction'] = 0.1
    lgb_params2['cat_cols'] = cat_cols
    results = run_cv_model(train, test, target, runLGB, lgb_params2, auc, 'lgb-lgb')


if RUN_LGB_W_FREQ:
    traintest = pd.concat([train, test])
    for col in cat_cols:
        print('Frequency Encoding {}'.format(col))
        traintest.loc[:, 'freq_{}'.format(col)] = traintest.groupby(col)[col].transform('count')
        traintest.drop(col, axis=1, inplace=True)
    train = traintest.iloc[:train.shape[0], :]
    test = traintest.iloc[train.shape[0]:, :]
    print(train.shape)
    print(test.shape)
    results = run_cv_model(train, test, target, runLGB, lgb_params, auc, 'lgb-freq')


if RUN_LGB_WITH_LR_ENCODING:
    for col in cat_cols:
        print('LR Encoding {}'.format(col))
        tr = pd.DataFrame(train[col])
        te = pd.DataFrame(test[col])
        tr, te = ohe(tr, te, col)
        print(tr.shape)
        print(te.shape)
        col_encode = run_cv_model(tr, te, target, runLR, lr_params, auc, 'lr-{}'.format(col))
        train.loc[:, 'lr_{}'.format(col)] = col_encode['train']
        test.loc[:, 'lr_{}'.format(col)] = col_encode['test']
        train.drop(col, axis=1, inplace=True)
        test.drop(col, axis=1, inplace=True)

    lgb_params2 = lgb_params.copy()
    lgb_params2['lambda_l1'] = 5
    lgb_params2['lambda_l2'] = 5
    results = run_cv_model(train, test, target, runLGB, lgb_params2, auc, 'lgb-lr')


if PRINT_LGB_FEATURE_IMPORTANCE:
    print_step('Feature importance')
    imports = results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()
    print(imports.sort_values('importance', ascending=False))


import pdb
pdb.set_trace()

print_step('Making submission')
submission = pd.DataFrame({'id': test_id, 'target': results['test']})
submission.to_csv('submission.csv', index=False)
