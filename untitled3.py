# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:09:23 2024

@author: alper
"""

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

veriseti = pd.read_excel('VeriOnIsleme.xlsx','Sayfa1')
x = veriseti.iloc[:,:,-1].values
y = veriseti.iloc[:,5].values

yaklasikdeger = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
yaklasikdeger = yaklasikdeger.fit(x[:,1,5])
x[:,1,6]=yaklasikdeger.trasform(x[:,1:5])

print(x)
print(yaklasikdeger.statistics_)