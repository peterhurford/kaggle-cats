## Base Models
* LR w/ Label Encoding - CV 0.7476 LB ?
* LGB w/ Freq Encoding - CV 0.7816 LB ?
* LGB w/ Label Encoding (old) - CV 0.7813 LB 0.78519 
* LGB w/ Label Encoding - CV 0.7817 LB ?
* Abhishek NN - CV 0.78887 LB 0.80576
* LGB w/ LGB Encoding - CV 0.80091 LB ?
* LR w/ OHE - CV 0.7982 LB ?
* LGB w/ LR encoding (old) - CV 0.80070 LB 0.80514
* LGB w/ LR encoding - CV 0.80203 LB ?
* Catboost - CV 0.80114 LB ?
* Target - CV 0.80115 LB ?
* LGB w/ Freq Encoding + All LR - CV 0.8025 LB ?
* LGB w/ LR encoding + All LR - CV 0.8031 LB ?
* DataRobot 100% AutoML (ENET w/Binning) - CV 0.8031 LB 0.80796
* LR w/ all OHE (old) - CV 0.8032 LB 0.80739
* LR w/ all OHE - CV 0.8033 LB ?
* LR w/ OHE + scalars + PL - CV 0.80362 LB ?
* LR w/ OHE + scalars (old) - CV 0.80364 LB 0.80795
* LR w/ OHE + Scalars + Suppress Rare (<10) - CV 0.80405 LB 0.80839
* LR w/ OHE + Scalars - CV 0.80406 LB 0.80833
* GLMNET w/ all OHE - CV 0.80412 LB ?
* LR w/ all OHE + scalars + targets - CV 0.80634 LB ?

## Base Blends
* 90%(LR w/ OHE + scalars (old)) + 10%(LGB w/ LR encoding) - CV 0.80367 LB 0.80795
* 85%(LR w/ OHE + scalars (old)) + 15%(Catboost) - CV 0.80372 LB ?

## 100 Fold Models
* GLMNET w/ all OHE; 100 folds - CV 0.80466 LB 0.80807
* LR w/ OHE + scalars; 100 folds (older) - CV 0.80477 LB 0.80807
* LR w/ OHE + scalars; 100 folds (old) - CV 0.80516 LB 0.80844
* LR w/ OHE + scalars; 100 folds - CV 0.80516 LB 0.80843

## 100 Fold Blends
* 80%(LR w/ OHE + scalars; 100 folds) + 20%(GLMNET w/ all OHE; 100 folds) - CV 0.80515 LB 0.80843
* 90%(LR w/ OHE + scalars; 100 folds) + 10%(GLMNET w/ all OHE; 100 folds) - CV 0.80516 LB 0.80844
