#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import re
import warnings
from IPython.display import Image
from util_reporting import (
    countplot_viz,
    boxplot_viz,
    histogram_viz,
    distplot_viz,
)

from util_data_cleaning import (
    extract_first_string,
    df_numeric_column_filler_with_aggregated_data,
)

from util_feature_engineering import (
    calculating_zscore,
)


get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')

# Seting a universal figure size<
rcParams["figure.figsize"] = 8, 6
df_Titanic = pd.read_csv("Titanic dataset.csv")
df_Titanic.head(5)


# In[29]:


sns.heatmap(df_Titanic.isnull(), cbar=False, yticklabels=False, cmap="mako")


# In[6]:


print("Count of non-missing rows of Age column:", df_Titanic["Age"].count())
print("Count of missing rows of Age column:", df_Titanic["Age"].isnull().sum())


# In[ ]:


df_Titanic["Appellation"] = df_Titanic["Name"].str.extract("([A-Za-z]+)\.")


# In[ ]:





# In[12]:


df_Titanic["Survived_"] = np.where(df_Titanic["Survived"] == 1, "survived", "died")


# In[13]:


df_Titanic["Survived_"].value_counts()


# In[14]:


countplot_viz(df_Titanic, "Survived_", "Survived & Died", "Count", "Count of Survived")


# In[16]:


df_Titanic.Pclass.value_counts()


# In[17]:


countplot_viz(df_Titanic, "Pclass", "Classes of passengers", "Count", "Count of Classes")


# In[19]:


list_of_column_descriptive = ["Age"]
df_descriptive_statistics(df_Titanic, list_of_column_descriptive)


# In[20]:


boxplot_viz(df_Titanic, "Age", xlabel="Age", title="Boxplot of Age")


# In[21]:


df_Titanic.Sex.value_counts()


# In[23]:


countplot_viz(df_Titanic, "Sex", "Gender of passengers", "Count", "Count of Genders")


# In[25]:


distplot_viz(
    data=df_Titanic,
    column="Age",
    separate_column="Survived_",
    condition_1="died",
    condition_2="survived",
    label1="died",
    label2="survived",
    title="Age Distribuition by Survived",
    color1="blue",
    color2="red",
)

