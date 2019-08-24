import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
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


print_step('Simple label encoding')
for c in train.columns:
    le = LabelEncoder()
    le.fit(pd.concat([train[c], test[c]])) 
    train[c] = le.transform(train[c])
    test[c] = le.transform(test[c])


print_step('OHE')
train_ohe, test_ohe = ohe(train, test)
print(train_ohe.shape)
print(test_ohe.shape)


print_step('Glimpsing')
print(train.head())


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
              'num_rounds': 10000}
results = run_cv_model(train_ohe.astype(np.float32), test_ohe.astype(np.float32), target, runLGB, lgb_params, auc, 'lgb-ohe')

print_step('Feature importance')
imports = results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()
print(imports.sort_values('importance', ascending=False))

print_step('Making submission')
submission = pd.DataFrame({'id': test_id, 'target': results['test']})
submission.to_csv('submission.csv', index=False)


lr_params = {'solver': 'lbfgs', 'C': 0.1151, 'scale': True}
results_lr = run_cv_model(train, test, target, runLR, lr_params, auc, 'lr-label')


lr_params = {'solver': 'lbfgs', 'C': 0.1151}
results_lr2 = run_cv_model(train_ohe, test_ohe, target, runLR, lr_params, auc, 'lr-ohe')


import pdb
pdb.set_trace()
