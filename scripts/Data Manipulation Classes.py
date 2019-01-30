
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Data-Manipulation-Classes" data-toc-modified-id="Data-Manipulation-Classes-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Data Manipulation Classes</a></span></li></ul></div>

# # Data Manipulation Classes
# 
# This notebooks collects the helper methods to be used along the different lessons. The two important classes 

# In[6]:


# imports

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

from src.dataset import Dataset

warnings.simplefilter(action='ignore')
warnings.filterwarnings(action='once')


# In[2]:


houses = Dataset('./data/houseprices_prepared.csv.gz')
houses.set_target('SalePrice')
houses.describe()


# Print a convenient table with the list of features that are categorical and contains NA. Other options are:
# 
#   - features
#   - target
#   - all
#   - complete
#   - numerical
#   - numerical_na
#   - categorical

# In[3]:


houses.table('categorical_na')


# Replace the NA's by new values in all 'categorical_na' features. There's a special case called 'Electrical' where NA is replaced by 'Unknown'. As you can see, you can pass a single column name or a list of column names.
# 
# To obtain a list of names from the dataset for each type of feature, we use `dataset.names(kind)`.

# In[4]:


houses.replace_na(column='Electrical', value='Unknown')
houses.replace_na(column=houses.names('categorical_na'), value='None')
houses.table('categorical_na')


# Describe now the dataset to check that there're no NA among the categorical variables!

# In[5]:


houses.describe()

