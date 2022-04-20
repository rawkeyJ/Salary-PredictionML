#!/usr/bin/env python
# coding: utf-8

# ### Importing the Modules

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression


# In[2]:


df = pd.read_csv('adult_salary.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df['workclass'] = df['workclass'].replace('?',np.nan)
df['occupation'] = df['occupation'].replace('?',np.nan)


# In[5]:


df.dropna(how='any',inplace=True)


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


df['income'].unique()


# ### Removing the Outliers

# In[9]:


sns.boxplot(df['hours-per-week'])


# In[10]:


def remove_outlier(df):
    IQR = df['hours-per-week'].quantile(0.75) - df['hours-per-week'].quantile(0.25) 
    
    lower_range = df['hours-per-week'].quantile(0.25) - (1.5 * IQR)
    upper_range = df['hours-per-week'].quantile(0.75) + (1.5 * IQR)
    
    df.loc[df['hours-per-week'] <= lower_range, 'hours-per-week'] = lower_range
    df.loc[df['hours-per-week'] >= upper_range, 'hours-per-week'] = upper_range


# In[11]:


remove_outlier(df)


# In[12]:


sns.boxplot(df['hours-per-week'])


# In[13]:


sns.boxplot(df['educational-num'])


# In[14]:


def remove_outlier_educationalnum(df):
    IQR = df['educational-num'].quantile(0.75) - df['educational-num'].quantile(0.25) 
    
    lower_range = df['educational-num'].quantile(0.25) - (1.5 * IQR)
    upper_range = df['educational-num'].quantile(0.75) + (1.5 * IQR)
    
    df.loc[df['educational-num'] <= lower_range, 'educational-num'] = lower_range
    df.loc[df['educational-num'] >= upper_range, 'educational-num'] = upper_range


# In[15]:


remove_outlier_educationalnum(df)


# In[16]:


sns.boxplot(df['educational-num'])


# In[17]:


sns.boxplot(df['capital-loss'])


# In[18]:


def remove_outlier_capitalloss(df):
    IQR = df['capital-loss'].quantile(0.75) - df['capital-loss'].quantile(0.25) 
    
    lower_range = df['capital-loss'].quantile(0.25) - (1.5 * IQR)
    upper_range = df['capital-loss'].quantile(0.75) + (1.5 * IQR)
    
    df.loc[df['capital-loss'] <= lower_range, 'capital-loss'] = lower_range
    df.loc[df['capital-loss'] >= upper_range, 'capital-loss'] = upper_range


# In[19]:


remove_outlier_capitalloss(df)


# In[20]:


sns.boxplot(df['capital-loss'])


# ### Plotting Correlation Graph

# In[21]:


plt.figure(figsize = (10, 10))
corr = df.corr()
sns.heatmap(corr, annot=True)


# ### Drop Columns based on the Correlation Graph

# In[22]:


df = df.drop('fnlwgt',axis=1)


# In[23]:


df.head()


# ### Initiating Feature Engineering 

# In[24]:


df['gender'] = df['gender'].map({'Male': 0, 'Female': 1}).astype(int)


# In[25]:


df.head()


# In[26]:


def feature_eng(df):
    clmtitle_enco_race = {value: key for key, value in enumerate(df['race'].unique())}
    df['race'] = df['race'].map(clmtitle_enco_race)
    
    clmtitle_enco_relationship = {value: key for key, value in enumerate(df['relationship'].unique())}
    df['relationship'] = df['relationship'].map(clmtitle_enco_relationship)
    
    clmtitle_enco_occupation = {value: key for key, value in enumerate(df['occupation'].unique())}
    df['occupation'] = df['occupation'].map(clmtitle_enco_occupation)
    
    clmtitle_enco_maritalstatus = {value: key for key, value in enumerate(df['marital-status'].unique())}
    df['marital-status'] = df['marital-status'].map(clmtitle_enco_maritalstatus)
    
    clmtitle_enco_education = {value: key for key, value in enumerate(df['education'].unique())}
    df['education'] = df['education'].map(clmtitle_enco_education)
    
    clmtitle_enco_workclass = {value: key for key, value in enumerate(df['workclass'].unique())}
    df['workclass'] = df['workclass'].map(clmtitle_enco_workclass)
    
    df['native-country'] = np.where(df['native-country'] == '?', 'Missing', df['native-country'])
    clmtitle_enco_nativecountry = {value: key for key, value in enumerate(df['native-country'].unique())}
    df['native-country'] = df['native-country'].map(clmtitle_enco_nativecountry)
    
    return df


# In[27]:


feature_eng(df)


# In[28]:


df.head()


# In[ ]:





# ### Initiating Scaling Process

# In[29]:


from sklearn.preprocessing import StandardScaler


# In[30]:


sc =  StandardScaler()


# In[32]:


X = df[['age', 'workclass', 'education','educational-num', 'marital-status', 'occupation', 'relationship',
        'race', 'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country']]


# In[33]:


y = df['income']


# In[34]:


y.value_counts()


# In[35]:


X =  sc.fit_transform(X)


# ### Train,Test and Model Building

# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[40]:


print("Train data shape: {}".format(X_train.shape))
print("Test data shape: {}".format(X_test.shape))


# In[41]:


lg_model = LogisticRegression()


# In[42]:


lg_model.fit(X_train, y_train)


# In[43]:


y_pred = lg_model.predict(X_test)


# In[44]:


result = {
    'Actual': y_test,
    'Predicted': y_pred
}


# In[45]:


pd.DataFrame(result)


# ### Model Accuracy and its Metrics

# In[46]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[48]:


print("Accuracy Score: {}".format(accuracy_score(y_test,y_pred)),"\n")
print("Confusion Matrix: {}".format(confusion_matrix(y_test,y_pred)),"\n")
print("Classification Report: {}".format(classification_report(y_test,y_pred)),"\n")


# ### Saving Model as File

# In[49]:


import pickle


# In[50]:


file = open('lg_model.pkl', 'wb')


# In[51]:


pickle.dump(lg_model, file)


# In[ ]:




