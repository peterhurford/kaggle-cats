STOP_AT_DATASET = False
RUN_LR_W_LABEL = False
RUN_LGB_W_FREQ = False
RUN_LGB_W_LABEL = False
RUN_LGB_W_LGB = False
RUN_LGB_BY_F = False
RUN_LGB_WITH_LR_ENCODING = False
RUN_LR_WITH_OHE = False
RUN_LR_WITH_ALL_OHE = False
RUN_LR_WITH_ALL_OHE_PLUS_SCALARS = True
RUN_TARGET = False

SUPPRESS_RARE = 10
ADD_LR = False
PRINT_LGB_FEATURE_IMPORTANCE = False
N_FOLDS = 100


import string

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score as auc

from utils import print_step, ohe, run_cv_model, runLGB, runFFLGB, runLR, runTarget


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


print_step('Combine')
traintest = pd.concat([train, test])


print_step('Cleaning ordinal')
ord_1 = ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster']
ord_1 = dict(zip(ord_1, range(len(ord_1))))
traintest.loc[:, 'ord_1'] = traintest['ord_1'].apply(lambda x: ord_1[x]).astype(int)

ord_2 = ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']
ord_2 = dict(zip(ord_2, range(len(ord_2))))
traintest.loc[:, 'ord_2'] = traintest['ord_2'].apply(lambda x: ord_2[x]).astype(int)

ord_3 = sorted(traintest['ord_3'].unique())
ord_3 = dict(zip(ord_3, range(len(ord_3))))
traintest.loc[:, 'ord_3'] = traintest['ord_3'].apply(lambda x: ord_3[x]).astype(int)

ord_4 = sorted(traintest['ord_4'].unique())
ord_4 = dict(zip(ord_4, range(len(ord_4))))
traintest.loc[:, 'ord_4'] = traintest['ord_4'].apply(lambda x: ord_4[x]).astype(int)

ord_5 = sorted(traintest['ord_5'].unique())
ord_5 = dict(zip(ord_5, range(len(ord_5))))
traintest.loc[:, 'ord_5'] = traintest['ord_5'].apply(lambda x: ord_5[x]).astype(int)


print_step('Transform day')
traintest['t_day'] = traintest['day'].apply(lambda d: np.abs(d - 4))


if SUPPRESS_RARE:
    print_step('Suppress rare categoricals ({})'.format(SUPPRESS_RARE))
    traintest = traintest.apply(lambda x: x.mask(x.map(x.value_counts()) < SUPPRESS_RARE, "rare"))


print_step('Re-split')
train = traintest.iloc[:train.shape[0], :]
test = traintest.iloc[train.shape[0]:, :]


if STOP_AT_DATASET:
    import pdb
    pdb.set_trace()


print_step('Simple label encoding')
cat_cols = [c for c in train.columns if 'nom' in c]
for c in cat_cols + ['bin_3', 'bin_4']:
    le = LabelEncoder()
    le.fit(pd.concat([train[c], test[c]])) 
    train.loc[:, c] = le.transform(train[c])
    test.loc[:, c] = le.transform(test[c])




lr_params = {'solver': 'liblinear', 'dual': True, 'C': 0.1151, 'max_iter': 1000}
if SUPPRESS_RARE:
    lr_params['C'] = 0.125


if RUN_LR_W_LABEL:
    lr_params2 = lr_params.copy()
    lr_params2['scale'] = True
    results = run_cv_model(train, test, target, runLR, lr_params2, auc, 'lr-label', n_folds=N_FOLDS)


if RUN_LR_WITH_ALL_OHE:
    print_step('OHE')
    train_ohe, test_ohe = ohe(train, test, train.columns)
    print(train_ohe.shape)
    print(test_ohe.shape)
    results_lr = run_cv_model(train_ohe, test_ohe, target, runLR, lr_params, auc, 'lr-all-ohe', n_folds=N_FOLDS)


if RUN_LR_WITH_ALL_OHE_PLUS_SCALARS:
    print_step('OHE')
    ohe_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
                'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',
                'month', 't_day']
    numeric_cols = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5',
                    'month', 't_day']
    train_ohe, test_ohe = ohe(train, test, cat_cols=ohe_cols, numeric_cols=numeric_cols)
    print(train_ohe.shape)
    print(test_ohe.shape)
    results = run_cv_model(train_ohe, test_ohe, target, runLR, lr_params, auc, 'lr-all-ohe-scalar', n_folds=N_FOLDS)


lgb_params = {'application': 'binary',
              'boosting': 'gbdt',
              'metric': 'auc',
              'num_leaves': 3,
              'max_depth': 1,
              'learning_rate': 0.05,
              'bagging_fraction': 0.9,
              'feature_fraction': 0.3,
              'verbosity': -1,
              'seed': 1,
              'lambda_l1': 0.1,
              'lambda_l2': 0.1,
              'early_stop': 100,
              'verbose_eval': 50,
              'num_rounds': 100000}


if ADD_LR:
    train.loc[:, 'lr'] = results['train']
    test.loc[:, 'lr'] = results['test']
    lgb_params['num_leaves'] = 4
    lgb_params['lambda_l2'] = 4
    lgb_params['max_depth'] = 2
    lgb_params['feature_fraction'] = 1.0


if RUN_LGB_W_LABEL:
    results = run_cv_model(train, test, target, runLGB, lgb_params, auc, 'lgb-label', n_folds=N_FOLDS)


if RUN_LGB_BY_F:
    # for col in cat_cols:
    #     print('LR Encoding {}'.format(col))
    #     tr = pd.DataFrame(train[col])
    #     te = pd.DataFrame(test[col])
    #     tr, te = ohe(tr, te, col)
    #     print(tr.shape)
    #     print(te.shape)
    #     col_encode = run_cv_model(tr, te, target, runLR, lr_params, auc, 'lr-{}'.format(col))
    #     train.loc[:, 'lr_{}'.format(col)] = col_encode['train']
    #     test.loc[:, 'lr_{}'.format(col)] = col_encode['test']
    #     train.drop(col, axis=1, inplace=True)
    #     test.drop(col, axis=1, inplace=True)
    lgb_params = {'application': 'binary',
                  'boosting': 'gbdt',
                  'metric': 'auc',
                  'num_leaves': 3,
                  'max_depth': 1,
                  'learning_rate': 0.1,
                  'bagging_fraction': 0.9,
                  'feature_fraction': 0.4,
                  'verbosity': -1,
                  'max_cat_to_onehot': 2,
                  'cat_smooth': 20,
                  'seed': 1,
                  'lambda_l1': 2,
                  'lambda_l2': 2,
                  'cat_cols': cat_cols}
                  # 'cat_cols': ['none']}
    results = run_cv_model(train, test, target, runFFLGB, lgb_params, auc, 'ff-lgb', n_folds=N_FOLDS)


if RUN_LGB_W_LGB:
    for col in train.columns:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
    lgb_params2 = lgb_params.copy()
    lgb_params2['max_cat_to_onehot'] = 2
    lgb_params2['cat_smooth'] = 20
    lgb_params2['num_leaves'] = 2
    lgb_params2['learning_rate'] = 0.03
    lgb_params2['lambda_l1'] = 2.0
    lgb_params2['lambda_l2'] = 2.0
    lgb_params2['feature_fraction'] = 0.1
    lgb_params2['cat_cols'] = cat_cols
    results = run_cv_model(train, test, target, runLGB, lgb_params2, auc, 'lgb-lgb', n_folds=N_FOLDS)


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
    results = run_cv_model(train, test, target, runLGB, lgb_params, auc, 'lgb-freq', n_folds=N_FOLDS)


if RUN_LGB_WITH_LR_ENCODING:
    for col in cat_cols:
        print('LR Encoding {}'.format(col))
        tr = pd.DataFrame(train[col])
        te = pd.DataFrame(test[col])
        tr, te = ohe(tr, te, col)
        print(tr.shape)
        print(te.shape)
        col_encode = run_cv_model(tr, te, target, runLR, lr_params, auc, 'lr-{}'.format(col), n_folds=N_FOLDS)
        train.loc[:, 'lr_{}'.format(col)] = col_encode['train']
        test.loc[:, 'lr_{}'.format(col)] = col_encode['test']
        train.drop(col, axis=1, inplace=True)
        test.drop(col, axis=1, inplace=True)

    lgb_params2 = lgb_params.copy()
    lgb_params2['lambda_l1'] = 5
    lgb_params2['lambda_l2'] = 5
    results = run_cv_model(train, test, target, runLGB, lgb_params2, auc, 'lgb-lr', n_folds=N_FOLDS)


if PRINT_LGB_FEATURE_IMPORTANCE:
    print_step('Feature importance')
    imports = results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()
    print(imports.sort_values('importance', ascending=False))


import pdb
pdb.set_trace()

print_step('Making submission')
submission = pd.DataFrame({'id': test_id, 'target': results['test']})
submission.to_csv('submission.csv', index=False)
