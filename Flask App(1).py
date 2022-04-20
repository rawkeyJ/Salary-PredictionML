#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, make_response, request, render_template


# In[2]:


import io
import pickle
from io import StringIO
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# In[3]:


app = Flask(__name__)


# In[ ]:


def feature_eng(df):
    df.columns = ['age', 'workclass','fnlwgt', 'education','educational-num', 'marital-status', 'occupation', 'relationship',
        'race', 'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    
    df = df.drop('fnlwgt',axis=1)
    
    df['gender'] = np.where(df['gender'] == "Male",1,0)
    
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

def scaler(df):
    sc = StandardScaler()
    X = df[['age', 'workclass', 'education','educational-num', 'marital-status', 'occupation', 'relationship',
        'race', 'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country']]
    X = sc.fit_transform(X)
    return(X)


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    f = request.files['data_file']
    if not f:
        return render_template('index.html', prediction_text="No file selected")
    
    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    result = stream.read()
    df = pd.read_csv(StringIO(result))
    
    
    df = feature_eng(df)
    
    X = scaler(df)
    
    loaded_model = pickle.load(open("lg_model.pkl","rb"))
    
    print(loaded_model)
    
    result = loaded_model.predict(X)
    
    return render_template('index.html',prediction_text="Predicted Salary is/are: {}".format(result))

if __name__ == "__main__":
    app.run(debug=False,port=9000)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




