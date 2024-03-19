# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:43:57 2024

@author: alper
"""

# import matplotlib.pyplot as plt

# plt.axis([2009,2015,30,80])
# plt.title("Türkiye'nin taşıma sektörüne ait sera gazı emisyon değerleri",fontsize=15,fontname='Cambria')
# plt.xlabel("Yıllar",fontsize=15,fontname='Cambria',color='red')
# plt.ylabel("Emisyon Değerleri (bin gram)",fontsize=15,fontname='Cambria')
# plt.text(2009.7,47,"En Düşük",color='Red',style='italic',weight='bold',size=9)
# plt.text(2013.5,74,"En Yüksek",color='Green')
# plt.grid()
# plt.plot([2010,2011,2012,2013,2014],[45,47,62,68,73],'b-')
# plt.plot([2010,2011,2012,2013,2014],[45,47,62,68,73],'ro')
# plt.legend(['Çizgi','Yuvarlak'],loc=8)
# plt.show()

# #%%
# import matplotlib.pyplot as plt
# import numpy as np
# Z=np.random.randn(100)
# X=Z*10+50
# plt.title("Makine Öğrenmesi Dersi Final Notların Histogramı")
# plt.xlabel('Notlar')
# plt.ylabel('Öğrenci Sayısı')
# plt.axis([0,100,0,25])
# plt.grid()
# n,bins,patches=plt.hist(X,bins=15,facecolor='blue',alpha=0.75)
# #%%
# import matplotlib.pyplot as plt
# import numpy as np
# from mp1_toolkits.mplot3d import Axes3D

# KumeA_x=np.random.randint(30,40,30)
# KumeA_y=np.random.randint(20,30,30)
# KumeA_z=np.random.randint(10,20,30)


# KumeB_x=np.random.randint(50,60,30)
# KumeB_y=np.random.randint(30,40,50)
# KumeB_z=np.random.randint(30,50,50)

# sekil=plt.figure()
# ax=Axes3D(sekil)
# #ax.scatter=(KumeA_x,KumeA_y,KumeA_z,c='g',marker='o')
# #ax.scatter=(KumeB_x,KumeB_y,KumeB_z,c='r',marker='^')

# ax.set_xlabel('X')
# ax.set_xlabel('Y')
# ax.set_xlabel('Z')
# plt.legend(['Küme A','Küme B'],loc=2)
#%%
#Özyinelemeli Özellik Eleme (RFE) yöntemi ile öznitelik seçimi 


from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
iris = datasets.load_iris()

model1 = LogisticRegression()

rfe = RFE(model1, n_features_to_select= 3)
rfe = rfe.fit(iris.data, iris.target)

print(rfe.support_)
print(rfe.ranking_)

from sklearn.svm import SVR
model2 = SVR(kernel="linear")

rfe = RFE(model2, n_features_to_select= 3)
rfe = rfe.fit(iris.data, iris.target)

print(rfe.support_)
print(rfe.ranking_)




















