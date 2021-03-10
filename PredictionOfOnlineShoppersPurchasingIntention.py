#!/usr/bin/env python
# coding: utf-8

# # Prediction of Online Shoppers Purchasing Intention - datasource.ai
# 
# In this competition, we will analyze the activity of users who vist a service/product offered online through a website. The objective is to predict which visitors will decide to buy according to the characteristics and interactions they exhibit on the site.
# 
# In this special case, we are working with a classification/clustering problem. Of the 12.330 sessions on the website, 84.58% did not decide to make a purchase, which equals 10.422 and the rest ended up buying (1908)

# The data set corresponds to 12,330 unique sessions per user, which are divided into
# 
#     8,631 for the training set (Train.csv)
#     3,699 for the test.csv set (Test.csv)
# 
# This data was obtained over 12 months to avoid special day trends or specific campaigns. 
# 
# In the file SampleSubmission.csv you can find the way in which you should send the data, and whose characteristics are:
# 
#     You must send your submission file with only 2 columns
#     Column 0 should be called: 'id’
#     Column 1 should be called: 'revenue’
#     The file must contain a total number of 3700 rows, where:
#         First row is == header
#         The other 3.699 rows == your predictions
#     If you do not meet these rules within your submission file, the system will automatically reject it
# 
# Note: we recommend you to check the file SampleSubmission.csv, which will be like this:
# 
# 
# id           revenue
#                   
# 1            0
# 2            0 
# 3 	     1
# 4            0 
# 5            1  
# 6            1
# etc.           
# 
# Variables definition:
# 
#     id: unique ID of the website visitor
#     administrative: Number of times the user visited the administrative section
#     administrative_duration: Total time the user spent in the administrative section
#     informational: Number of times the user visited the informational section
#     informational_duration: Total time the user spent in the informational section
#     productrelated: Number of times the user visited the related products section
#     productrelated_duration: Total time the user spent in the related products section
#     bouncerates: This is the percentage of visitors who enter the page and immediately "bounce" without interacting with it. This metric is only taken into account if it is the first page they visit within the website.
#     exitrates: From the total number of visits to the pages of the website, the percentage of visitors who logged out through this specific page is obtained, that is, it indicates the percentage of users whose last visit to the website was this specific page.
#     pagevalues: This is the average value of the website, it indicates the contribution that this website made to the visitor arriving at the final purchase page or section. 
#     specialday: Is the value that indicates the proximity to a special date such as Valentine's Day.  The range of this variable is 0 to 1, with 1 being the exact day of the special date and 0 if there is no range near that date.
#     month: Month of the visit to the website.
#     operatingsystems: Type of operating system
#     browser: Name of the web browser
#     region: Visitor's geographic region
#     traffictype: Type of web traffic
#     visitortype: Whether you are a new visitor or a returning visitor
#     Weekend: 0 indicates that it is not a weekend day and 1 indicates that it is a weekend day.
# 
# Target variable:
# 
#     revenue: Variable to be classified, 1 indicates that the visitor has bought and 0 indicates that the visitor has not bought.

# In[4]:


import re
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

sns.set(style='dark', context='notebook', palette='plasma')


# In[9]:


path = Path('C:\\Users\\desar\\OneDrive\\Escritorio\\data_science\\Prediction of Online Shoppers Purchasing Intention')

train = pd.read_csv(path/'train_set.csv', index_col = 'id')
test = pd.read_csv(path/'test_set.csv', index_col = 'id')


# In[10]:


train.head()


# # EDA
# 
# El dataset ya esta limpio

# In[11]:


print(train.shape)
print(test.shape)


# In[13]:


print(train.describe())


# In[17]:


print(train.info())


# In[27]:


corr_matrix = train.corr()

_= plt.figure(figsize = (15,7))
_= plt.title('Matriz de correlacion - train set')
_= sns.heatmap(corr_matrix, cmap = 'viridis', annot = True)

plt.show()


# In[28]:


corr_matrix['revenue'].sort_values()


# In[105]:


#Cut outliers for better visulization

train_v = train.copy()
train_v.loc[train.Administrative_Duration > 1000, 'Administrative_Duration'] = 1000
train_v.loc[train.ProductRelated_Duration > 10000, 'ProductRelated_Duration'] = 10000
train_v.loc[train.Informational_Duration > 100, 'Informational_Duration'] = 100

#Plot visualization

_= plt.figure(figsize = (22,7))
_= plt.title('Adm duration vs Related Products duration')
_= plt.xlabel('Administrative Duration')
_= plt.ylabel('Realted Products Duration')
_= plt.scatter(train_v.Administrative_Duration, train_v.ProductRelated_Duration, 
               train_v.Informational_Duration, alpha = .3)


# In[113]:


#Swarmplot takes really long to run when many values overlap, make violinplot instead

_= plt.figure(figsize = (10,5))
_= sns.violinplot(train_v['revenue'], train_v['ProductRelated_Duration'], hue =  train_v['VisitorType'], cut = 0)
plt.show()
_= plt.figure(figsize = (10,5))
_= sns.violinplot(train_v['revenue'], train_v['Administrative_Duration'], hue =  train_v['VisitorType'])
plt.show()
_= plt.figure(figsize = (10,5))
_= sns.violinplot(train_v['revenue'], train_v['Informational_Duration'], hue =  train_v['VisitorType'])
plt.show()


# In[125]:


_= plt.figure(figsize = (10,5))
_= sns.barplot('Administrative', 'Administrative_Duration', hue = 'revenue', data = train, ci = False)
plt.show()
_= plt.figure(figsize = (10,5))
_= sns.barplot('Informational', 'Informational_Duration', hue = 'revenue', data = train, ci = False)
plt.show()


# In[151]:


_= sns.catplot('ProductRelated', 'ProductRelated_Duration', data=train_v, height=5, aspect=2)
#_= plt.xticks(x, "")     #Delete x axis labels for cleannes
plt.show()


# In[173]:


_= sns.barplot(train.revenue, train.SpecialDay, train.Weekend)
plt.show()


# In[175]:


_= plt.figure(figsize = (10,5))
_= sns.violinplot(train.OperatingSystems, train.Region)


# In[184]:


_= plt.figure(figsize = (20,10))
_= sns.barplot(train.revenue, train.PageValues, train.Month)


# In[189]:


_= plt.figure(figsize = (15,7))
_= sns.scatterplot(train.BounceRates, train.ExitRates, train.revenue, alpha =.5)


# In[199]:


_= plt.figure(figsize = (10,5))
_= sns.barplot('Month', 'revenue', data = train, ci = False)
plt.show()


# JUEVES: VER DE SACAR ALGUN P-VALUE! A LA TARDE Primer modelo ya
