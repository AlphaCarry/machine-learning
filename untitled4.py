# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:19:45 2024

@author: alper
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

veriseti = pd.read_excel('VeriOnIsleme.xlsx','Sayfa1')

X = veriseti.iloc[:,:-1].values
y = veriseti.iloc[:,5].values

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

print(X)