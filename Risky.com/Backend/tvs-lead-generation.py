#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import sklearn
print(sklearn.__version__)


# **Read Data**

# In[ ]:


df_temp=pd.read_csv("./TVS.csv",low_memory="false")
df_temp


# In[ ]:


import sklearn.feature_selection as fs
x=df_temp.iloc[:,0:31]
y=pd.DataFrame(df_temp["V32"])
x,y


# In[ ]:


x=x.drop(["V1"],axis=1)
x


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(df_temp.isnull(), cbar=False)


# In[ ]:


x.describe()


# **Null Imputation**

# In[ ]:


x["V10"].mode()


# In[ ]:


x["V14"]=x["V14"].fillna("Z")


# In[ ]:


x["V10"]=x["V10"].fillna("SC")


# In[ ]:


list_V10=x["V10"].unique()
sorted(list_V10)


# In[ ]:


list_V14=x["V14"].unique()
sorted(list_V14)


# In[ ]:


x["V15"]=x["V15"].fillna("NO")


# In[ ]:


list_V15=x["V15"].unique()
sorted(list_V15)


# In[ ]:


x=x.drop(['V16'],axis=1)


# In[ ]:


x["V17"]=x["V17"].fillna(x["V17"].mean())


# In[ ]:


x["V21"]=x["V21"].fillna(0)


# In[ ]:


x=x.drop(["V21","V22","V23","V24","V26","V27"],axis=1)


# In[ ]:


x["V25"]=x["V25"].fillna(x["V25"].mean())


# In[ ]:


sns.heatmap(x.isnull(), cbar=False)


# **Label Encoding**

# In[ ]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
  
x["V31"]=label_encoder.fit_transform(x["V31"]) #Tier1=0,Tier2=1,Tier3=2
print(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
x["V10"]=label_encoder.fit_transform(x["V10"]) #Motorcycle-0,Moped-1,Retop-2,Scooter-3,Tl-4
print(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
x["V13"]=label_encoder.fit_transform(x["V13"]) #Female-0,Male-1
print(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
x["V14"]=label_encoder.fit_transform(x["V14"]) #Houswife-0,Pension-1,Salaried-2,Self-3,Student-4,Unemployed-5
print(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
x["V15"]=label_encoder.fit_transform(x["V15"]) #Others-0,Office-Owned-1,Owned-2,Rent-3
print(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))


# In[ ]:


x["V4"]=x["V4"].fillna(x["V4"].mean())
x["V5"]=x["V5"].fillna(x["V5"].mean())
x["V6"]=x["V6"].fillna(x["V6"].mean())
x["V7"]=x["V7"].fillna(x["V7"].mean())
x["V8"]=x["V8"].fillna(x["V8"].mean())
x["V9"]=x["V9"].fillna(x["V9"].mean())
x["V11"]=x["V11"].fillna(x["V11"].mean())
x["V12"]=x["V12"].fillna(x["V12"].mean())
x["V13"]=x["V13"].fillna(x["V13"].mean())


# In[ ]:


dfvis= x.join(y)
dfvis


# **Correlation**

# In[ ]:


sns.set (rc = {'figure.figsize':(20, 20)})
dataplot = sns.heatmap(dfvis.corr(), cmap="YlGnBu", annot=True)
dataplot


# In[ ]:


from sklearn.model_selection import train_test_split
x=x.loc[:,["V3","V5","V17","V28","V29","V30"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
uni = SelectKBest(score_func = f_classif, k = 6)
fit = uni.fit(x, y)
features=x.columns[fit.get_support(indices=True)].tolist()
features


# **Lazy-Predict**

# ![add.png](attachment:d3e4ab17-79da-46b6-8127-a853f548e3a9.png)

# **Quadratic Discriminant Analysis**

# In[ ]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
quad = QuadraticDiscriminantAnalysis()
quad.fit(x_train, y_train)


# In[ ]:


from sklearn.metrics import accuracy_score  
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
y_pred1=quad.predict(x_test)
accuracy=accuracy_score(y_test, y_pred1)
roc_auc=roc_auc_score(y_test,y_pred1)
f1=f1_score(y_test, y_pred1,average='weighted')
accuracy,roc_auc,f1


# **Random Forest Classifier and Hyperparameter Tuning**

# In[ ]:


x.shape


# In[ ]:


from sklearn.ensemble import RandomForestClassifier  
scores =[]
for i in range(1,502,25):
    rfc = RandomForestClassifier(n_estimators=i)
    rfc.fit(x_train, y_train)
    y_pred_temp_estimators = rfc.predict(x_test)
    scores.append(accuracy_score(y_test, y_pred_temp_estimators))
    print("Accuracy with estimators = "+str(i)+" is = "+str(accuracy_score(y_pred_temp_estimators,y_test)))


# In[ ]:


sns.set (rc = {'figure.figsize':(15, 6)})
plt.plot(range(1,502,25), scores)
plt.xlabel('Value of n_estimators for Random Forest Classifier')
plt.ylabel('Testing Accuracy')


# In[ ]:


scores2=[]
for i in range(1,11):
    rfc = RandomForestClassifier(n_estimators=425,max_depth=i)
    rfc.fit(x_train, y_train)
    y_pred_temp_estimators2 = rfc.predict(x_test)
    scores2.append(accuracy_score(y_test, y_pred_temp_estimators2))
    print("Accuracy with estimators = "+str(i)+" is = "+str(accuracy_score(y_pred_temp_estimators2,y_test)))


# In[ ]:


plt.plot(range(1,11), scores2)
plt.xlabel('Value of max_depth for Random Forest Classifier')
plt.ylabel('Testing Accuracy')


# In[ ]:


classifier_rfc=RandomForestClassifier(n_estimators=475,max_depth=7)
classifier_rfc.fit(x_train,y_train)


# In[ ]:


y_pred_rfc=classifier_rfc.predict(x_test)
rfc_acc=accuracy_score(y_pred_rfc,y_test)
rfc_acc


# **Dump**

# In[ ]:


import pickle
pickle.dump(classifier_rfc,open("RandomForest_6Params.pkl","wb"))
pickle.dump(quad,open("QuadraticDiscriminant_6Params.pkl","wb"))


# In[ ]:




