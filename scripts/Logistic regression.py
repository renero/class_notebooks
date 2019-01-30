
# coding: utf-8

# # Logistic Regression
# 
# Logistic Regression is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the logistic regression model predicts P(Y=1) as a function of X.
# 
# It works very much the same way Linear Regression does, except that the optimization function is not OLS but [_maximum likelihood_](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation).

# ## Important considerations
# 
# - We use logistic regression to train a model to predict between 2-classes: Yes/No, Black/White, True/False. If we need to predict more than two classes, we need to build some artifacts in logistic regression that will be explained at the end of this notebook.
# - No dependent variables should be among the set of features. Study the correlation between all the features separatedly.
# - Scaled, norm'd and centered input variables.
# 
# The output from a logistic regression is always the log of the odds. We will explore this concept further along the exercise.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm

from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from matplotlib.gridspec import GridSpec


# In[2]:


data = pd.read_csv('train_complete_prepared.csv.gz')
data.head()


# Lets build a dataframe that will contain the type and number of NAs that each feature contains. We will use it to decide what variables to select. We will now from there what features are numerical and categorical, and how many contain NAs.

# In[3]:


def dataframe_metainformation(df):
    meta = dict()
    descr = pd.DataFrame({'dtype': df.dtypes, 'NAs': df.isna().sum()})
    categorical_features = descr.loc[descr['dtype'] == 'object'].index.values.tolist()
    numerical_features = descr.loc[descr['dtype'] != 'object'].index.values.tolist()
    numerical_features_na = descr.loc[(descr['dtype'] != 'object') & (descr['NAs'] > 0)].index.values.tolist()
    categorical_features_na = descr.loc[(descr['dtype'] == 'object') & (descr['NAs'] > 0)].index.values.tolist()
    complete_features = descr.loc[descr['NAs'] == 0].index.values.tolist()
    meta['description'] = descr
    meta['categorical_features'] = categorical_features
    meta['categorical_features'] = categorical_features
    meta['categorical_features_na'] = categorical_features_na
    meta['numerical_features'] = numerical_features
    meta['numerical_features_na'] = numerical_features_na
    meta['complete_features'] = complete_features
    return meta

def print_metainformation(meta):
    print('Available types:', meta['description']['dtype'].unique())
    print('{} Features'.format(meta['description'].shape[0]))
    print('{} categorical features'.format(len(meta['categorical_features'])))
    print('{} numerical features'.format(len(meta['numerical_features'])))
    print('{} categorical features with NAs'.format(len(meta['categorical_features_na'])))
    print('{} numerical features with NAs'.format(len(meta['numerical_features_na'])))
    print('{} Complete features'.format(len(meta['complete_features'])))


# In[4]:


meta = dataframe_metainformation(data)
print_metainformation(meta)


# #### Can we build a model that will predict the contents of one of those categorical columns with NAs?
# 
# Let's try! I will start with `FireplaceQu` that presents a decent amount of NAs.
# 
# Define **target** and **features** to hold the variable we want to predict and the features I can use (those with no NAs). We remove the `Id` from the list of features to be used by our model. Finally, we establish what is the source dataset, by using only those rows from `data` that are not equal to NA.
# 
# Lastly, we will encode all categorical features (but the target) to have a proper setup for running the logistic regression. To encode, we'll use OneHotEncoding by calling `get_dummies`. The resulting dataset will have all numerical features.

# In[5]:


target = 'FireplaceQu'
features = meta['complete_features']
features.remove('Id')
print('Selecting {} features'.format(len(features)))

data_complete = data.filter(features + [target])
data_complete = data_complete[data_complete[target].notnull()]

meta_complete = dataframe_metainformation(data_complete)
print_metainformation(meta_complete)
dummy_columns = meta_complete['categorical_features']
dummy_columns.remove(target)
data_encoded = pd.get_dummies(data_complete, columns=dummy_columns)
data_encoded.head(3)


# How many occurences do we have from each class of the target variable?

# In[6]:


sns.countplot(x='FireplaceQu', data=data_encoded);
plt.show();


# Since we've very few occurences of classes `Ex`, `Fa` and `Po`, we will remove them from the training set, and we will train our model to learn to classify only between `TA` or `Gd`.

# In[7]:


data_encoded = data_encoded[(data_encoded[target] != 'Ex') & 
                            (data_encoded[target] != 'Fa') & 
                            (data_encoded[target] != 'Po')]
data_encoded[target] = data_encoded[target].map({'TA':0, 'Gd':1})
sns.countplot(x='FireplaceQu', data=data_encoded);


# Set the list of features prepared

# In[8]:


features = list(data_encoded)
features.remove(target)


# ### Recursive Feature Elimination
# Recursive Feature Elimination (RFE) is based on the idea to repeatedly construct a model and choose either the best or worst performing feature, setting the feature aside and then repeating the process with the rest of the features. This process is applied until all features in the dataset are exhausted. The goal of RFE is to select features by recursively considering smaller and smaller sets of features.

# In[9]:


from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

X = data_encoded.loc[:, features]
y = data_encoded.loc[:, target]

logreg = LogisticRegression(solver='lbfgs', max_iter=250)
rfe = RFE(logreg, 15)
rfe = rfe.fit(X, y)

print('Selected features: {}'.format(list(data_encoded.loc[:, rfe.support_])))


# ## Building the model
# 
# Set the variables $X$ and $Y$ to the contents of the dataframe I want to use, and fit a `Logit` model. Print a summary to check the results. We're using the `statmodels` package because we want easy access to all the statistical indicators that logistic regression can lead to.

# In[10]:


X = data_encoded.loc[:, list(data_encoded.loc[:, rfe.support_])]
y = data_encoded.loc[:, target]

logit_model=sm.Logit(y, X)
result=logit_model.fit(method='bfgs')
print(result.summary2())


# ### P-Values and feature selection
# 
# Remove those predictors with _p-values_ above 0.05
# 
# Mark those features with a p-value higher thatn 0.05 (or close) to be removed from $X$, and run the logistic regression again to re-.check the p-values. From that point we'll be ready to run the model properly in sklearn.

# In[11]:


to_remove = result.pvalues[result.pvalues > 0.05].index.tolist()
X.drop(to_remove, inplace=True, axis=1)

logit_model=sm.Logit(y, X)
result=logit_model.fit(method='bfgs')
print(result.summary2())


# ### The Logit model
# 
# Here we train the model and evaluate on the test set. The interpretation of the results obtained by calling the `classification_report` are as follows:
# 
# The **precision** is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier to not label a sample as positive if it is negative.
# 
# The **recall** is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
# 
# The **F-beta** score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.
# 
# The F-beta score weights the recall more than the precision by a factor of beta. beta = 1.0 means recall and precision are equally important.
# 
# The **support** is the number of occurrences of each class in y_test.

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=0)
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy on test: {:.2f}'.format(logreg.score(X_test, y_test)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ### ROC Curve
# 
# The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers. The dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner).

# In[13]:


logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])


# Plot the FPR vs. TPR, and the diagonal line representing the null model.

# In[14]:


def plot_roc(fpr, tpr, logit_roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show();


# In[15]:


plot_roc(fpr, tpr, logit_roc_auc)


# The results are very poor, and what we've got shouldn't be used in production. The proposal from this point is:
#   1. to know more about how the predictions are made in logistic regression
#   2. apply a logit to predict if the price of a house will be higher or lower than a given value

# ### Explore logit predictions
# 
# What you've seen is that we irectly call the method `predict` in `logit`, which will tell me to which class each sample is classified: 0 or 1. To accomplish this, the model produces two probabilities

# In[16]:


pred_proba_df = pd.DataFrame(logreg.predict_proba(X_test))
threshold_list = np.arange(0.05, 1.0, 0.05)
accuracy_list = np.array([])
for threshold in threshold_list:
    y_test_pred = pred_proba_df.applymap(lambda prob: 1 if prob > threshold else 0)
    test_accuracy = accuracy_score(y_test.values,
                                   y_test_pred[1].values.reshape(-1, 1))
    accuracy_list = np.append(accuracy_list, test_accuracy)


# And the plot of the array of accuracy values got from each of the probabilities.

# In[17]:


plt.plot(range(accuracy_list.shape[0]), accuracy_list, 'o-', label='Accuracy')
plt.title('Accuracy for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.xticks([i for i in range(1, accuracy_list.shape[0], 2)], 
           np.round(threshold_list[1::2], 1))
plt.grid()
plt.show();


# ## Default Dataset
# 
# A simulated data set containing information on ten thousand customers. The aim here is to predict which customers will default on their credit card debt. A data frame with 10000 observations on the following 4 variables.
# 
# `default`
#     A factor with levels No and Yes indicating whether the customer defaulted on their debt
#     
# `student`
#     A factor with levels No and Yes indicating whether the customer is a student
#     
# `balance`
#     The average balance that the customer has remaining on their credit card after making their monthly payment
#     
# `income`
#     Income of customer
# 

# In[18]:


data = pd.read_csv('default.csv')
data.head()


# Let's build a class column with the proper values on it (0 and 1) instead of the strings with Yes and No.

# In[19]:


data.default = data.default.map({'No': 0, 'Yes': 1})
data.student = data.student.map({'No': 0, 'Yes': 1})
data.head()


# We are interested in predicting whether an individual will default on his or her credit card payment, on the basis of annual income and monthly credit card balance.
# 
# It is worth noting that figure below displays a very pronounced relationship between the predictor balance and the response default. In most real applications, the relationship between the predictor and the response will not be nearly so strong.

# In[20]:


def plot_descriptive(data):
    fig = plt.figure(figsize=(9, 4))
    gs = GridSpec(1, 3, width_ratios=[3, 1, 1])

    ax0 = plt.subplot(gs[0])
    ax0 = plt.scatter(data.balance[data.default==0], 
                      data.income[data.default==0], 
                      label='default=No',
                      marker='.', c='red', alpha=0.5)
    ax0 = plt.scatter(data.balance[data.default==1], 
                      data.income[data.default==1],
                      label='default=Yes',
                      marker='+', c='green', alpha=0.7)
    ax0 = plt.xlabel('balance')
    ax0 = plt.ylabel('income')
    ax0 = plt.legend(loc='best')
    ax0 = plt.subplot(gs[1])
    ax1 = sns.boxplot(x="default", y="balance", data=data)
    ax0 = plt.subplot(gs[2])
    ax2 = sns.boxplot(x="default", y="income", data=data)

    plt.tight_layout()
    plt.show()


# In[21]:


plot_descriptive(data)


# Consider again the Default data set, where the response `default` falls into one of two categories, Yes or No. Rather than modeling this response $Y$ directly, logistic regression models the probability that $Y$ belongs to a particular category.
# 
# For example, the probability of default given balance can be written as
# 
# $$Pr(default = Yes|balance)$$
# 
# The values of $Pr(default = Yes|balance)$ –$p(balance)$–,  will range between 0 and 1. Then for any given value of `balance`, a prediction can be made for `default`. For example, one might predict `default = Yes` for any individual for whom $p(balance) > 0.5$.

# In[22]:


def plot_classes(show=True):
    plt.scatter(data.balance[data.default==0], 
                data.default[data.default==0], 
                marker='o', color='red', alpha=0.5)
    plt.scatter(data.balance[data.default==1], 
                data.default[data.default==1], 
                marker='+', color='green', alpha=0.7)
    plt.xlabel('Balance')
    plt.ylabel('Probability of default')
    plt.yticks([0, 1], [0, 1])
    if show is True:
        plt.show();


# In[23]:


plot_classes()


# Build the model, and keep it on `logreg`.

# In[24]:


X_train, X_test, y_train, y_test = train_test_split(data.balance, 
                                                    data.default,
                                                    test_size=0.3, 
                                                    random_state=0)
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train.values.reshape(-1, 1), y_train)

y_pred = logreg.predict(X_test.values.reshape(-1, 1))
acc_test = logreg.score(X_test.values.reshape(-1, 1), y_test)

print('Accuracy on test: {:.2f}'.format(acc_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# And now represent where is the model setting the separation function between the two classes.

# In[25]:


def plot_sigmoid():
    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    plot_classes(show=False)
    plt.plot(sigm.x.values, sigm.y.values, color='black', linewidth=3);
    plt.title('Sigmoid')

    plt.subplot(1, 2, 2)
    plot_classes(show=False)
    plt.plot(sigm.x.values, sigm.y.values, color='black', linewidth=3);
    plt.xlim(1925, 1990)
    plt.title('Zooming the Sigmoid')
    plt.tight_layout()
    plt.show()


# In[26]:


def model(x):
    return 1 / (1 + np.exp(-x))

y_func = model(X_test.values * logreg.coef_ + logreg.intercept_).ravel()
sigm = pd.DataFrame({'x': list(X_test.values), 'y': list(y_pred)}).                    sort_values(by=['x'])
plot_sigmoid()


# ## Next steps
# 
# - Explore multinomial logistic regression with sklearn
# - Explore SKLearn pipelines to find the optimal parameters of Logit
# - Explore Lasso and Ridge Regression with Linear Regression problems
# - Continue reading by exploring GLM (Generalized Linear Models).
