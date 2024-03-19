# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:30:06 2024

@author: alper
"""

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

Immunotherapy =pd.read_excel('Immunotherapy1.xlsx')

model = LogisticRegression()

rfe = RFE(model,n_features_to_select= 3)
rfe = rfe.fit(Immunotherapy.iloc[:,0:7],Immunotherapy.iloc[:,7])

print(rfe.support_)
print(rfe.ranking_)

#%%

from sklearn.svm import SVR
estimator = SVR(kernel= 'linear')
rfe = RFE(estimator,n_features_to_select= 3)

rfe = rfe.fit(Immunotherapy.iloc[:,0:7],Immunotherapy.iloc[:,7])

print(rfe.support_)
print(rfe.ranking_)