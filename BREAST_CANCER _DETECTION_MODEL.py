#!/usr/bin/env python
# coding: utf-8

# ### Importing Dependencies

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Data Collection and Analysis

# In[3]:


from sklearn.datasets import load_breast_cancer
cancer_dataset=load_breast_cancer()
print(cancer_dataset)

#finding the keyvalues from dictionary
print(cancer_dataset.keys())
type(cancer_dataset)


# In[4]:


print(cancer_dataset['feature_names'])


# In[5]:


#creating a dataframe
can_df=pd.DataFrame(np.c_[cancer_dataset["data"],cancer_dataset["target"]],
                   columns=np.append(cancer_dataset["feature_names"],["target"]))
can_df


# In[6]:


can_df.to_csv("breast_cancer_dataframe.csv")
can_df.head()   #will print first five values of the dataframe


# In[7]:


can_df.info()


# In[8]:


can_df.describe() #describe the data in detail form


# ### Data Visualization

# In[9]:


sns.countplot(can_df["target"])


# In[10]:


sns.countplot(can_df["mean radius"]) #mean radius more then 1 having cancer


# In[11]:


plt.figure(figsize=(15,15))
sns.heatmap(can_df)
#


# In[12]:


plt.figure(figsize=(15,15))
sns.heatmap(can_df.corr(),annot=True,cmap="hot",linewidth=3)


# In[13]:


X=can_df.drop(["target"],axis=1)
y=can_df["target"]


# In[14]:


from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42) 
print(X_train)
print(X_test)
print(y_train) 
print(y_test)


# In[15]:


from sklearn.preprocessing import StandardScaler 
sc=StandardScaler() 
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
X_train  


# ## Model Training

# ### Support vector classifier

# In[16]:


from sklearn.svm import SVC
classifier=SVC() 
classifier.fit(X_train,y_train) 


# In[17]:


y_pred=classifier.predict(X_test)
y_pred 


# In[18]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_pred,y_test) 
print(cm)
print(accuracy_score(y_pred,y_test) )
classification_report(y_pred,y_test)


# ### Logistic regression

# In[19]:


from  sklearn.linear_model import LogisticRegression 
lg_classifier=LogisticRegression(random_state=0)
lg_classifier.fit(X_train,y_train)


# In[20]:


y_pred=lg_classifier.predict(X_test)
y_pred 


# In[21]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_pred,y_test) 
print(cm) 
accuracy_score(y_pred,y_test)


# ### K - Nearest Neighbor

# In[22]:


from sklearn.neighbors import KNeighborsClassifier
KN_classifier=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2) 
KN_classifier.fit(X_train,y_train) 


# In[23]:


y_pred=KN_classifier.predict(X_test)
y_pred 


# In[24]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_pred,y_test) 
print(cm) 
accuracy_score(y_pred,y_test) 


# ### Naive Bias classifier

# In[25]:


from sklearn.naive_bayes import GaussianNB 
NB_classifier=GaussianNB()
NB_classifier.fit(X_train,y_train) 


# In[26]:


y_pred=NB_classifier.predict(X_test) 
y_pred  


# In[27]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_pred,y_test) 
print(cm) 
accuracy_score(y_pred,y_test) 


# ### Decision tree classifier

# In[28]:


from sklearn.tree import DecisionTreeClassifier
D_classifier=DecisionTreeClassifier(criterion="entropy",random_state=0) 
D_classifier.fit(X_train,y_train) 


# In[29]:


y_pred =D_classifier.predict(X_test)
y_pred 


# In[30]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_pred,y_test) 
print(cm) 
accuracy_score(y_pred,y_test)


# ### Random forest classifier

# In[31]:


from sklearn.ensemble import RandomForestClassifier 
rc_classifier=RandomForestClassifier(n_estimators=20,criterion="entropy",random_state=0)
rc_classifier.fit(X_train,y_train) 


# In[32]:


y_pred=rc_classifier.predict(X_test)
y_pred 


# In[33]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_pred,y_test) 
print(cm) 
accuracy_score(y_pred,y_test) 


# In[34]:


from sklearn.model_selection import cross_val_score 
cross_validation=cross_val_score(estimator=rc_classifier,X=X_train,y=y_train) 
cross_validation
print("cross validation mean accuracy",cross_validation.mean()) 


# In[35]:


import pickle 
pickle.dump(rc_classifier,open("breast_cancer.pickle","wb"))
breast_cancer_model=pickle.load(open("breast_cancer.pickle","rb"))
y_pred=breast_cancer_model.predict(X_test)
print(confusion_matrix(y_pred,y_test))
print(accuracy_score(y_pred,y_test)) 


# In[ ]:




