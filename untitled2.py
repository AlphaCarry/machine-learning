# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:51:00 2024

@author: alper
"""

import matplotlib.pyplot as plt #noqa: F401

from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

digits = load_digits()
x = digits.images.reshape((len(digits.images),-1))
y = digits.target

svc = SVC(kernel='linear',C=1)
rfe = RFE(estimator=svc,n_features_to_select=1,step=1)
rfe.fit(x,y)

print(rfe.support_)
print(rfe.ranking_)

ranking = rfe.ranking_.reshape(digits.images[0].shape)
plt.matshow(ranking.cmap==plt.cm.Blues)
plt.colorbar()
plt.matshow()