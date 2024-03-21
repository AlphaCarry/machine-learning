# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:28:29 2024

@author: alper
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

veriseti = pd.read_excel('VeriOnIsleme_2.xlsx','Sayfa1')

X1 = veriseti.iloc[:,0].values
y = veriseti.iloc[:,5].values

labelencoder_X = LabelEncoder()
X1 = labelencoder_X.fit_transform(X1)
X1 = X1.reshape(-1, 1)

onehotencoder_X = OneHotEncoder(categories='auto')
X2 = onehotencoder_X.fit_transform(X1).toarray()
print(X2)

new = np.zeros((14,7))
yy = veriseti.iloc[:,1:5].values

new[:,0:3] = X2
new[:,3:7] = yy

print(new)