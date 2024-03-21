# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:41:48 2024

@author: alper
"""

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

veriseti = pd.read_excel('VeriOnIsleme_2.xlsx','Sayfa1')

X = veriseti.iloc[:,:-1].values
y = veriseti.iloc[:,5].values

onehotencoder_X = ColumnTransformer(
    [('one_hot_encoder',OneHotEncoder(categories='auto'),[0])],remainder='passthrough'
    )
      
X2 = onehotencoder_X.fit_transform(X)
print(X2)