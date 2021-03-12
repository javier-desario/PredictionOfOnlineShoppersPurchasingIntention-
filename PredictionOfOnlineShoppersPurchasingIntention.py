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

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from pathlib import Path

sns.set(style='dark', context='notebook', palette='plasma')
random_state = 42


# In[2]:


path = Path('C:\\Users\\desar\\OneDrive\\Escritorio\\data_science\\Prediction of Online Shoppers Purchasing Intention')

train = pd.read_csv(path/'train_set.csv', index_col = 'id')
test = pd.read_csv(path/'test_set.csv', index_col = 'id')


# In[3]:


train.head()


# # EDA
# 
# El dataset ya esta limpio

# In[4]:


train.shape, test.shape


# In[5]:


train.describe()


# In[6]:


train.info()


# In[7]:


corr_matrix = train.corr()

_= plt.figure(figsize = (15,7))
_= plt.title('Matriz de correlacion - train set')
_= sns.heatmap(corr_matrix, cmap = 'viridis', annot = True)

plt.show()


# In[8]:


corr_matrix['revenue'].sort_values()


# In[9]:


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


# In[10]:


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


# In[11]:


_= plt.figure(figsize = (10,5))
_= sns.barplot('Administrative', 'Administrative_Duration', hue = 'revenue', data = train, ci = False)
plt.show()
_= plt.figure(figsize = (10,5))
_= sns.barplot('Informational', 'Informational_Duration', hue = 'revenue', data = train, ci = False)
plt.show()


# In[12]:


_= sns.catplot('ProductRelated', 'ProductRelated_Duration', data=train_v, height=5, aspect=2)
_= plt.xticks([])     #Delete x axis labels for cleannes
plt.show()


# In[13]:


_= sns.barplot(train.revenue, train.SpecialDay, train.Weekend)
plt.show()


# In[14]:


_= plt.figure(figsize = (10,5))
_= sns.violinplot(train.OperatingSystems, train.Region)


# In[15]:


_= plt.figure(figsize = (20,10))
order = ['Ene', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
_= sns.barplot(train.revenue, train.PageValues, train.Month, hue_order=order)


# In[16]:


_= plt.figure(figsize = (15,7))
_= sns.scatterplot(train.BounceRates, train.ExitRates, train.revenue, alpha =.5)


# In[17]:


train.Month.unique()


# In[18]:


_= plt.figure(figsize = (10,5))
order = ['Ene', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
_= sns.barplot('Month', 'revenue', data = train, ci = False, order=order)
plt.show()


# In[19]:


train.plot(kind='scatter', x='revenue', y='PageValues');


# In[20]:


_= plt.figure(figsize = (15,7))
_= sns.scatterplot(train.ExitRates, train.PageValues, train.revenue, alpha =.5)


# # Preprocessing

# Preprocessing de las variables VisitorType, Weekend & Month

# In[21]:


train.head()


# In[22]:


train.VisitorType.unique()


# In[23]:


VisitorType_map = {'New_Visitor': 0, 'Returning_Visitor': 1, 'Other': 2}
train.VisitorType = train.VisitorType.map(VisitorType_map)
test.VisitorType = test.VisitorType.map(VisitorType_map)


# In[24]:


def WeekendMapping(x):
    """Maps Weekend bool var to binary var"""
    
    if x:
        return 1
    else:
        return 0

for df in [train, test]:
    df['WeekendOk'] = df.Weekend.apply(WeekendMapping)  
    df.drop('Weekend', axis=1, inplace=True)


# In[25]:


ns = [i for i in range(len(train.Month.unique()))]
months = train.Month.unique().tolist()

MonthMapping = dict(zip(months, ns))
for df in [train, test]:
    df.Month = df.Month.map(MonthMapping)


# In[26]:


test.head()


# In[27]:


X = train.copy()
X_test = test.copy()
X.drop('revenue', axis=1, inplace=True)
y = train['revenue']

X_train, X_valid, y_train, y_valid = train_test_split(
                                        X, y, test_size=0.2, 
                                        random_state=random_state)

print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape, X_test.shape)


# # Modelling

# In[28]:


knc = KNeighborsClassifier()
knc_model = knc.fit(X_train, y_train)
y_pred = knc_model.predict(X_valid)

f1_score(y_valid, y_pred, average = 'macro')


# In[29]:


logreg = LogisticRegression()
logreg_model = logreg.fit(X_train, y_train)
y_pred = logreg_model.predict(X_valid)

f1_score(y_valid, y_pred, average = 'macro')


# In[30]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda_model = lda.fit(X_train, y_train)
y_pred = lda.predict(X_valid)

f1_score(y_valid, y_pred, average = 'macro')


# In[31]:


svc = SVC(random_state=random_state, probability = True)
svc_model = svc.fit(X_train, y_train)
y_pred = svc_model.predict(X_valid)

f1_score(y_valid, y_pred, average = 'macro')


# In[32]:


tree = DecisionTreeClassifier(random_state=random_state)
tree_model = tree.fit(X_train, y_train)
y_pred = tree_model.predict(X_valid)

f1_score(y_valid, y_pred, average = 'macro')


# In[33]:


forest = RandomForestClassifier(random_state=random_state)
rfc_model = forest.fit(X_train, y_train)
y_pred = rfc_model.predict(X_valid)

f1_score(y_valid, y_pred, average = 'macro')


# In[34]:


etc = ExtraTreesClassifier(random_state=random_state)
etc_model = forest.fit(X_train, y_train)
y_pred = etc_model.predict(X_valid)

f1_score(y_valid, y_pred, average = 'macro')


# In[35]:


gbc = GradientBoostingClassifier(random_state=random_state)
gbc_model = gbc.fit(X_train, y_train)
y_pred = gbc_model.predict(X_valid)

f1_score(y_valid, y_pred, average = 'macro')


# In[36]:


abc = AdaBoostClassifier(random_state=random_state)
abc_model = abc.fit(X_train, y_train)
y_pred = abc_model.predict(X_valid)

f1_score(y_valid, y_pred, average = 'macro')


# In[37]:


mlpc = MLPClassifier(random_state=random_state)
mlpc_model = mlpc.fit(X_train, y_train)
y_pred = mlpc_model.predict(X_valid)

f1_score(y_valid, y_pred, average = 'macro')


# In[40]:


#rf_param_grid = {"max_depth": [None],
              #"max_features": [1, 7, 15],
              #"min_samples_split": [2, 5, 15],
              #"min_samples_leaf": [1, 3, 10],
              #"bootstrap": [False],
              #"n_estimators" :[100,500],
              #"criterion": ["gini"]}

#kfold = StratifiedKFold(n_splits=4)
#gsRFC = GridSearchCV(forest, param_grid = rf_param_grid, cv=kfold, scoring="f1", n_jobs= -1, verbose = 1)
#gsRFC.fit(X_train,y_train)
#RFC_best = gsRFC.best_estimator_
# Best score
#gsRFC.best_score_

#y_pred = gsRFC.predict(X_valid)
#f1_score(y_valid, y_pred, average = 'macro')


# In[44]:


#gsRFC.best_params_


# In[46]:


votingC = VotingClassifier(
    estimators=[('rfc', RandomForestClassifier(max_features=7,
                                               min_samples_leaf=3,
                                               min_samples_split=15,
                                               n_estimators=500, 
                                               random_state = random_state)),
                ('extc', ExtraTreesClassifier(random_state = random_state)),                 
                ('gbc', GradientBoostingClassifier(random_state = random_state))],
    voting = 'hard', n_jobs = 6)

voting_model = votingC.fit(X_train, y_train)
y_pred = voting_model.predict(X_valid)

f1_score(y_valid, y_pred, average = 'macro')


# In[39]:


sub = gbc_model.predict(X_test)
submission = pd.DataFrame({'id': test.index, 'revenue': sub})
submission.to_csv('submission.csv', index = False)

