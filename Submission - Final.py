#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-info">
# <h1> TELECOM CHURN - CASE STUDY</h1>
# 
# <h2>Submitted by: Praveersinh Parmar & Ketaki Samanta</h2>
# </div>

# # <font color=LimeGreen>✍️ Problem Statement </font>

# ## <font color=SlateBlue>Business problem overview</font>
# 
# - In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, **customer retention** has now become even more important than customer acquisition.
# 
#  
# 
# - For many incumbent operators, *retaining high profitable customers is the number one business goal*.
# 
#  
# 
# - To reduce customer churn, telecom companies need to **predict which customers are at high risk of churn**.
# 
#  
# 
# - In this project, you will analyse customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn and identify the main indicators of churn.

# ## <font color=SlateBlue>Understanding and defining churn</font>
# 
# 
# - There are two main models of payment in the telecom industry - 
# 
#   1. **Postpaid** (customers pay a monthly/annual bill after using the services) and 
#   2. **Prepaid** (customers pay/recharge with a certain amount in advance and then use the services).
# 
#  
# 
# - In the ***postpaid model***, when customers want to switch to another operator, they usually inform the existing operator to terminate the services, and you directly know that this is an instance of churn.
# 
#  
# 
# - However, in the ***prepaid model***, customers who want to switch to another network can simply stop using the services without any notice, and it is hard to know whether someone has actually churned or is simply not using the services temporarily (e.g. someone may be on a trip abroad for a month or two and then intend to resume using the services again).
# 
#  
# 
# - Thus, churn prediction is usually more critical (and non-trivial) for prepaid customers, and the term ‘`churn`’ should be defined carefully.    
# - Also, prepaid is the most common model in *India* and *Southeast Asia*, while postpaid is more common in *Europe* and *North America*.
# 
#  
# 
# - This project is based on the *Indian* and *Southeast Asian* market.

# ## <font color=SlateBlue>Definitions of churn</font>
# 
# - There are various ways to define churn, such as:
# 
# > **Revenue-based churn**: Customers who have not utilised any revenue-generating facilities such as mobile internet, outgoing calls, SMS etc. over a given period of time. One could also use aggregate metrics such as ‘`customers who have generated less than INR 4 per month in total/average/median revenue`’.
#   - The main shortcoming of this definition is that there are customers who only receive calls/SMSes from their wage-earning counterparts, i.e. they don’t generate revenue but use the services. For example, many users in rural areas only receive calls from their wage-earning siblings in urban areas.
# 
# 
# 
# 
# > **Usage-based churn**: Customers who have not done any usage, either incoming or outgoing - in terms of calls, internet etc. over a period of time.
#   - A potential shortcoming of this definition is that when the customer has stopped using the services for a while, it may be too late to take any corrective actions to retain them. For e.g., if you define churn based on a ‘two-months zero usage’ period, predicting churn could be useless since by that time the customer would have already switched to another operator.
# 
#  
# 
# In this project, you will use the **usage-based definition** to define churn.

# ## <font color=SlateBlue>High-value churn</font>
# 
# - In the Indian and the Southeast Asian market, approximately 80% of revenue comes from the top 20% customers (called **high-value customers**). Thus, if we can reduce churn of the high-value customers, we will be able to reduce significant revenue leakage.
# 
#  
# 
# - In this project, you will define high-value customers based on a certain metric (mentioned later below) and predict churn only on high-value customers.

# ## <font color=SlateBlue>Understanding the business objective and the data</font>
# 
# - The dataset contains customer-level information for a span of **four consecutive months** - `June`, `July`, `August` and `September`. The months are encoded as `6`, `7`, `8` and `9`, respectively. 
# 
# 
# - The business objective is `to predict the churn in the last (i.e. the ninth) month using the data (features) from the first three months`. To do this task well, understanding the typical customer behaviour during churn will be helpful.

# ## <font color=SlateBlue>Understanding customer behaviour during churn</font>
# 
# - Customers usually do not decide to switch to another competitor instantly, but rather over a period of time (this is especially applicable to high-value customers). In churn prediction, we assume that there are **three phases** of customer lifecycle :
# 
#    1. The ‘**good**’ phase: In this phase, the customer is happy with the service and behaves as usual.
# 
#    2. The ‘**action**’ phase: The customer experience starts to sore in this phase, for e.g. he/she gets a compelling offer from a  competitor, faces unjust charges, becomes unhappy with service quality etc. In this phase, the customer usually shows different behaviour than the ‘good’ months. Also, it is crucial to identify high-churn-risk customers in this phase, since some corrective actions can be taken at this point (such as matching the competitor’s offer/improving the service quality etc.)
# 
#    3. The ‘**churn**’ phase: In this phase, the customer is said to have churned. You **define churn based on this phase**. Also, it is important to note that at the time of prediction (i.e. the action months), this data is not available to you for prediction. Thus, **after tagging churn as 1/0 based on this phase, you discard all data corresponding to this phase**.
# 
# 
# 
# In this case, since you are working over a four-month window
# 
# - the *first two months* are the ‘**good**’ phase, 
# - the *third month* is the ‘**action**’ phase, 
# - while the *fourth month* is the ‘**churn**’ phase.
# 
#  

# ## <font color=SlateBlue>Data Preparation</font>
# 
# The following data preparation steps are crucial for this problem:
# 
#  
# 
# 1. **Derive new features**
# 
#    - This is one of the most important part of data preparation since good features are often the differentiators between good and bad models. 
#    - Use your business understanding to derive features you think could be important indicators of churn.
# 
#  
# 
# 2. **Filter high-value customers**
# 
#    - As mentioned above, you need to predict churn only for the high-value customers. 
#    - Define **high-value customers** as follows: Those who have recharged with an amount more than or equal to X, where X is the **70th percentile** of the average recharge amount in the first two months (the good phase).
#    - After filtering the high-value customers, you should get **about 29.9k** rows.
# 
#  
# 
# 3. **Tag churners and remove attributes of the churn phase**
# 
# Now tag the churned customers (churn=1, else 0) based on the fourth month as follows: 
#   - Those who have not made any calls (either incoming or outgoing) AND have not used mobile internet even once in the churn phase. 
#   
# The attributes you need to use to tag churners are:
# 
# - total_ic_mou_9
# 
# - total_og_mou_9
# 
# - vol_2g_mb_9
# 
# - vol_3g_mb_9
# 
# 
# After tagging churners, **remove all the attributes corresponding to the churn phase** (all attributes having ‘ _9’, etc. in their names).
# 
#  

# ## <font color=SlateBlue>Modelling</font>
# 
# Build models to predict churn. The predictive model that you’re going to build will serve two purposes:
# 
# 1. It will be used **to predict whether a high-value customer will churn or not, in near future (i.e. churn phase)**. By knowing this, the company can take action steps such as providing special plans, discounts on recharge etc.
# 
# 2. It will be used **to identify important variables that are strong predictors of churn**. These variables may also indicate why customers choose to switch to other networks.
# 
#  
# 
# - In some cases, both of the above-stated goals can be achieved by a single machine learning model. But here, you have a large number of attributes, and thus you should try using a **dimensionality reduction technique such as PCA** and then build a predictive model. **After PCA, you can use any classification model**.
# 
#  
# 
# - Also, since the rate of churn is typically low (about 5-10%, this is called class-imbalance) - try using techniques to handle class imbalance. 
# 
#  
# 
# You can take the following suggestive **steps to build the model**:
# 
# 1. Preprocess data (convert columns to appropriate formats, handle missing values, etc.)
# 
# 2. Conduct appropriate exploratory analysis to extract useful insights (whether directly useful for business or for eventual modelling/feature engineering).
# 
# 3. Derive new features.
# 
# 4. Reduce the number of variables using PCA.
# 
# 5. Train a variety of models, tune model hyperparameters, etc. (handle class imbalance using appropriate techniques).
# 
# 6. Evaluate the models using appropriate evaluation metrics. Note that it is more important to identify churners than the non-churners accurately - choose an appropriate evaluation metric which reflects this business goal.
# 
# 7. Finally, choose a model based on some evaluation metric.
# 
#  

# - The above model will only be able to achieve one of the two goals - **to predict customers who will churn**. You can’t use the above model to identify the important features for churn. That’s because PCA usually creates components which are not easy to interpret.
# 
#  
# 
# - Therefore, **build another model with the main objective of identifying important predictor attributes which help the business understand indicators of churn**. A good choice to identify important variables is a logistic regression model or a model from the tree family. In case of logistic regression, make sure to handle multi-collinearity.
# 
#  
# 
# - After identifying important predictors, display them visually - you can use plots, summary tables etc. - whatever you think best conveys the importance of features.
# 
#  
# 
# - Finally, recommend strategies to manage customer churn based on your observations.
# 

# # <font color=lightcoral>🌊 Flow of this notebook</font>
# 
# The following are steps involved in problem solving:
# 1. **Data Understanding & Preparation**
#     - Data Quality Checks
#     - Data Cleaning
#     - Filtering High Value Customers
#     - Tagging Churned Customers
#     - Feature Engineering
#     - Exploratory Data Analysis (EDA)
#     - Outlier Treatment
#     
#    
# 2. **Model Building and Evaluation # 1 : Interpretable Model**  
#     (This model will identify important predictors of churn)
# 
#    - Dimensionality Reduction (using RFE)
#    - Model Building
#    - Hyperparameter Tuning
#    - Model Evaluation
#    - Identifying important churn predictors
#    - Business recommendations
#    
# 
# 3. **Model Building and Evaluation # 2 : High Performance Models**  
#     (These models will predict aacurately the customers who will churn)
#     
#    - Dimensionality Reduction (using PCA)
#    - Three models: Logistic Regression, Random Forest and XGBoost
#    - Model Building
#    - Hyperparameter Tuning
#    - Model Evaluation   
#    
#    
# 4. **Summarizing all the models created and selecting the best one**

# In[1]:


## Suppress warnings

import warnings
warnings.filterwarnings("ignore")


# ## 📚 Importing Libraries

# In[2]:


## Import requisite libraries

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('fivethirtyeight')

import datetime as dt

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.metrics import recall_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

from statsmodels.stats.outliers_influence import variance_inflation_factor

import xgboost as xgb


# In[3]:


## Set limits for displaying rows and columns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# # <font color=midnightblue>Step-1️:</font>  🤔 <font color=mediumvioletred>Data Understanding & Preparation</font>

# In[4]:


## Import the telecom churn data set

telecom = pd.read_csv("telecom_churn_data.csv")

## See the first five rows of our dataframe

telecom.head()


# In[5]:


## Read the Data Dictionary excel file

dict = pd.read_excel('Data+Dictionary-+Telecom+Churn+Case+Study.xlsx')
dict


# In[6]:


## Check the dimensions of telecom dataset
print(f"Number of Rows = {telecom.shape[0]} \nNumber of Columns = {telecom.shape[1]}")


# ### 📌  The `telecom` dataframe has 99999 rows and 226 columns

# # <font color=steelblue>✅ Data Quality Checks</fonts>

# In[7]:


## Check the column-wise info

telecom.info(verbose=True)


# In[8]:


## See the statistical description of numerical columns

telecom.describe(percentiles=[.25,.5,.75,.90,.95,.99])


# ### 📌  There are few columns like `circle_id` that contain only one value. As there is no variation in data, these columns are of no use to us and we should drop them

# In[9]:


## Drop columns having only one value in all rows
for col in telecom.columns:
    if telecom[col].nunique() == 1:
        telecom.drop(col,inplace=True,axis=1)


# In[10]:


## Check the dimensions to see how many columns have been reduced
telecom.shape


# ### 📌 Here, we have removed 226 - 210 = 16 columns that were not useful to us for making predictions

# In[11]:


## Checking data types of columns
telecom.dtypes


# ### 📌 We need to convert eight date columns from 'object' to 'datetime' type

# In[12]:


## List out the eight object columns
date_cols = list(telecom.select_dtypes(include='object').columns)
date_cols


# In[13]:


## Convert these eight columns to datetime format
for col in date_cols:
    telecom[col] = pd.to_datetime(telecom[col], infer_datetime_format=True)


# In[14]:


## Check the converted data types
telecom.dtypes


# # <font color=steelblue>🔎 Data Cleaning</fonts>

# In[15]:


## Check percentage of missing values in descending order
nulls = round(100*(telecom.isna().sum())/(len(telecom.index)),2).sort_values(ascending=False)
nulls


# In[16]:


## Extract features having more than 70% null values
nulls = list(nulls[nulls >= 70].index)
nulls


# In[17]:


## Drop the features having more than 70% missing values as they would not be much useful in making predictions
telecom = telecom.drop(nulls, axis=1)

## Check dimensions again
telecom.shape


# ### 📌 So, now we have removed 210 - 170 = 40 columns that were having more than 70% missing values

# In[18]:


## Check the percentage of null values again
nulls = round(100*(telecom.isna().sum())/(len(telecom.index)),2).sort_values(ascending=False)
nulls


# In[19]:


## Retain only the columns with non-zero missing values
nulls = nulls[nulls > 0]
nulls


# In[20]:


## Check the types of columns that have missing values
telecom[nulls.index].dtypes


# ### 📌 Except, the four datetime columns, we can replace the missing values of all other columns with their median values. We use mean here as there are outliers.
# 
# ### 📌 For datetime columns, missing values are less than 5% of the data. 
# 

# In[21]:


## Drop the four datetime columns from nulls
nulls.drop(['date_of_last_rech_6', 'date_of_last_rech_7', 'date_of_last_rech_8','date_of_last_rech_9'], inplace=True)


# In[22]:


## Replace the missing values of the remaining columns with their median values
for col in nulls.index:
    telecom[col].fillna(telecom[col].median(), inplace=True)


# In[23]:


## Check again for percentage of missing values
nulls = round(100*(telecom.isna().sum())/(len(telecom.index)),2).sort_values(ascending=False)
nulls


# ### 📌 Now, we are only left with missing values in four date columns. We will impute them using method 'ffill' that will just replace it with the last observed value.

# In[24]:


## Create a list of four date columns
date_cols = ['date_of_last_rech_6', 'date_of_last_rech_7', 'date_of_last_rech_8', 'date_of_last_rech_9']


# In[25]:


## Impute the missing values in these four columns using method 'ffill'
for col in date_cols:
    telecom[col].fillna(method='ffill', inplace=True)


# In[26]:


## Check again for the missing values
telecom.isna().sum()


# ### 📌So, out dataset is cleaned now. 

# # <font color=steelblue>💰 Filtering High Value Customers</fonts>

# In[27]:


## Create new column for 'average recharge amount in first two months' (good phase)
telecom['avg_rech_good_phase'] = (telecom['total_rech_amt_6'] + telecom['total_rech_amt_7'])/2
telecom['avg_rech_good_phase'].head()


# In[28]:


## Check the dimensions again (one columns must have increased)
telecom.shape


# ### 📌 Now, we extract those customers who have recharged with an amount more than or equal to 70th percentile of average recharge of first two months (i.e. good phase)

# In[29]:


## Extracting 'High Value Customers'
telecom = telecom[telecom['avg_rech_good_phase'] >= telecom['avg_rech_good_phase'].quantile(0.7)]


# In[30]:


## Checking the dimensions again
telecom.shape


# ### 📌 So, we have now kept only about 30k rows of data pertaining to `High Value Customers`

# # <font color=steelblue>🏷️ Tagging Churned Customers</fonts>

# In[31]:


## Now we create new column 'churn' where we tag customers who have not made any calls and 
## not used any mobile internet in 9th month

telecom['churn'] = np.where( (telecom['total_ic_mou_9']==0) & (telecom['total_og_mou_9']==0 ) &                             (telecom['vol_2g_mb_9']==0) & (telecom['vol_3g_mb_9']==0) , 1, 0)
telecom.head()


# ### 📌 The purpose of all 9th months columns was to tag customers as 'churn' (1) or 'non-churn'(0). 
# ### 📌 Also, it is important to note that at the time of prediction (i.e. the action months), this churn data is not available to us for prediction. Thus, we now discard all data corresponding to this phase, i.e. we remove all columns ending with '_9'.

# In[32]:


## Removing all columns ending with '_9'
for col in telecom.columns:
    if str(col).endswith('_9'):
        telecom.drop(col, axis=1, inplace=True)
telecom.head()


# In[33]:


## Check the dimensions again
telecom.shape 


# ### 📌 Now, we are left with 131 features.

# In[34]:


## Check the imbalance in target variable 'churn'
telecom.churn.value_counts(normalize=True)


# In[35]:


## View the histogram and pie-chart to see distribution of 'churn' variable
plt.figure(figsize=(15,7))

# Pie-chart
plt.subplot(121)
telecom["churn"].value_counts().plot.pie(autopct = "%1.0f%%", wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.1,0],shadow =True)
plt.title("Distribution of CHURN variable")

# Histogram
plt.subplot(122)
sns.countplot(x='churn', data=telecom, hue='churn')
plt.title("Count of CHURN variable");


# ### 📌 So, our target variable is highly imbalanced.
# ### 📌 We will handle this imbalance while building models.

# # <font color=steelblue>✨ Feature Engineering</fonts>

# In[36]:


## View the remaining date columns
telecom.select_dtypes(include='datetime').head()


# ### 📌 Now, we will create new features from these three date columns giving the number of days past since last recharge, for the months  6,7 and 8.

# In[37]:


## Now we create new features for 'number of days since last recharge'

today = dt.date.today()
telecom['days_since_last_rech_6'] = (pd.to_datetime(today) - telecom['date_of_last_rech_6']).dt.days
telecom['days_since_last_rech_7'] = (pd.to_datetime(today) - telecom['date_of_last_rech_7']).dt.days
telecom['days_since_last_rech_8'] = (pd.to_datetime(today) - telecom['date_of_last_rech_8']).dt.days

## View the newly formed columns
telecom.head()


# In[38]:


## Now, we can drop the three original datetime columns
telecom.drop(['date_of_last_rech_6', 'date_of_last_rech_7', 'date_of_last_rech_8'], axis=1, inplace=True)
telecom.head()


# In[39]:


## Check the dimensions
telecom.shape


# ### 📌 We created 3 new columns and removed 3 columns, so number of columns is the same: 131

# # <font color=steelblue>🧐 Exploratory Data Analysis (EDA)</fonts>

# In[40]:


## Extract numerical columns, i.e. all columns except mobile number column
df_num = telecom.drop(['mobile_number'], axis=1)
df_num.head()


# # 🕐 Univariate Analysis

# In[41]:


## View the distribution of all variables (except four datetime columns) using histograms and boxplots

for col in df_num.columns:
    plt.figure(figsize=(15,5))
    
    # Plot histogram with kde
    plt.subplot(1,2,1)
    plt.title(col, fontdict={'fontsize': 18})
    sns.distplot(df_num[col])
    
    # Plot boxplot
    plt.subplot(1,2,2)
    sns.boxplot(df_num[col])
    plt.show()    


# ### 📌 So, we see that almost all the numeric features have outliers.So, we will have to do outlier treatment before building models.
# ### 📌 Some numerical features have skewed distributions.

# # 🕑 Bivariate Analysis

# In[42]:


## View the correlations among numerical variables
plt.figure(figsize=(25,25))
sns.heatmap(df_num.corr(), cmap='coolwarm');


# ### 📌 We observe that there is multicollinearity among the numerical variables. We will have to deal with it while building models.

# # <font color=steelblue>💹 Outlier Treatment</fonts>

# In[43]:


## Check the boxplots to view outliers in numerical columns
plt.figure(figsize=[25,25])
plt.xticks(rotation=90)
sns.boxplot(data=df_num);


# In[44]:


## Capping outliers to 5% at lower bound and 95% at upper bound
for col in df_num.columns:
    telecom[col][telecom[col] <= telecom[col].quantile(0.05)] = telecom[col].quantile(0.05)
    telecom[col][telecom[col] >= telecom[col].quantile(0.95)] = telecom[col].quantile(0.95)


# In[45]:


## Check the boxplots again to see if outliers have been treated
plt.figure(figsize=[25,10])
plt.xticks(rotation=90)
sns.boxplot(data=telecom);


# ### 📌 So, we have removed outliers from all numerical columns

# # <font color=midnightblue>Step-2:</font>  🏗️ <font color=mediumvioletred>Model Building and Evaluation # 1 : Interpretable Model</font>

# ### First, we will build a logistic regression model with the objective of identifying important predictor attributes which help the business understand indicators of churn.

# # 🔠 Create feature and target variables: X and y

# In[46]:


## Create the Feature Variable
X = telecom.drop(['mobile_number', 'churn'], axis=1)
X.head()


# In[47]:


## Create the Target Variable 
y = telecom["churn"]
y.head()


# # ✂️Splitting Data into Training and Test Sets

# In[48]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, stratify=y, random_state=42)


# In[49]:


## Check the dimensions of train and test data
print(f"Shape of X_train = ({X_train.shape} \nShape of y_train = ({y_train.shape} \nShape of X_test = ({X_test.shape} \nShape of y_test = ({y_test.shape}")


# # 📏 Scaling the data

# In[50]:


## Scale the feature variable
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## View snippet of scaled training data and X_test
print(f"X_train (snippet) \n{X_train[:5, :5]} \nX_test (snippet) \n {X_test[:5, :5]}")


# In[51]:


## View types of X_train, X_test, y_train and y_test
print(f"X_train:{type(X_train)} \nX_test:{type(X_test)}  \ny_train:{type(y_train)}  \ny_test:{type(y_test)}")


# In[52]:


## Convert X_train and X_test from numpy array to pandas dataframe
X_train = pd.DataFrame(X_train, columns = X.columns)
X_test = pd.DataFrame(X_test, columns = X.columns)


# In[53]:


## View types of X_train and X_test again
print(f"X_train:{type(X_train)} \nX_test:{type(X_test)}")


# # <font color=steelblue>📉 Dimensionality Reduction (using RFE)</fonts>

# In[54]:


## Instantiate a Logistic Regression Model
#######################################################################################
## Here, we set parameter class_weight to 'balanced' in order to HANDLE CLASS IMBALANCE
#######################################################################################

logreg = LogisticRegression(random_state=42, n_jobs=-1, class_weight='balanced')


# In[55]:


## Run RFE with 15 variables as output
rfe = RFE(estimator=logreg, n_features_to_select=15, verbose=1, step=5)             
rfe = rfe.fit(X_train, y_train)


# In[56]:


## List out the 15 features selected by RFE
temp = pd.Series(rfe.support_,index = X.columns)
selected_features_rfe = list(temp[temp==True].index)
print(selected_features_rfe)


# In[57]:


## Keep only these 15 features in X_train and X_test
X_train_rfe = X_train[selected_features_rfe]
X_test_rfe = X_test[selected_features_rfe]


# # ♊ Handling Multicollinearity

# In[58]:


## Creating a helper function in which we create a dataframe that will contain names  
## of all feature variables and their respective VIFs

def show_vifs(X_train):
    vif = pd.DataFrame()
    vif['Features'] = X_train_rfe.columns
    vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    print(vif)
    return


# In[59]:


## Display the VIF values of all variables of X_train_rfe
show_vifs(X_train_rfe)


# In[60]:


## Drop the variable with highest VIF : 'loc_og_mou_8'
X_train_rfe.drop('loc_og_mou_8', axis=1, inplace=True)

## View VIF values again
show_vifs(X_train_rfe)


# In[61]:


## Drop the variable with highest VIF : 'loc_ic_mou_8'
X_train_rfe.drop('loc_ic_mou_8', axis=1, inplace=True)

## View VIF values again
show_vifs(X_train_rfe)


# In[62]:


## Drop the variable with highest VIF : 'total_og_mou_8'
X_train_rfe.drop('total_og_mou_8', axis=1, inplace=True)

## View VIF values again
show_vifs(X_train_rfe)


# ### 📌 Now, all the variables left have VIF values below 5. So, we don't need to drop any more features. 

# In[63]:


## We will also drop the above variables from X_test_rfe
X_test_rfe.drop(['loc_og_mou_8', 'loc_ic_mou_8', 'total_og_mou_8'], axis=1, inplace=True)


# # <font color=steelblue>🧰 Model Building</fonts>

# ### Choosing the evaluation metric
# 📌 Since in our model, it is more important to identify churners than non-churners:
#    - it is important for us to correctly identify positives (churners) that are actual postives i.e. `TP should be high`. 
#    - it is important that we do not identify a positive(churner) as negative(non-churner) i.e. `FN should be low`.  
#    
# 📌 So, here we will select **Recall (or Sensitivity)** as our evluation metric which is given by 
# ###               $Recall = \frac{TP}{TP + FN}$

# In[64]:


## Build a baseline logistic regression model with only 15 features selected by RFE
logreg_rfe_baseline = LogisticRegression(random_state=42, n_jobs=-1, class_weight='balanced')


# In[65]:


## Train the Logistic Regression model on training data containing 15 selected features
logreg_rfe_baseline.fit(X_train_rfe, y_train)


# In[66]:


## Make predictions on the test data
y_pred = logreg_rfe_baseline.predict(X_test_rfe)
y_pred


# In[67]:


## View predicted probabilities
preds = logreg_rfe_baseline.predict_proba(X_test_rfe)
preds


# In[68]:


## View the confusion matrix
print(confusion_matrix(y_test, y_pred))

fig, ax = plt.subplots(figsize=(15,8))
plot_confusion_matrix(logreg_rfe_baseline, X_test_rfe, y_test, display_labels=['Not_Churn', 'Churn'], ax=ax);


# In[69]:


## View the recall score
recall_rfe_baseline = recall_score(y_test, y_pred)
recall_rfe_baseline = round(recall_rfe_baseline, 2)
recall_rfe_baseline


# ### 📌 The Recall score we are getting in this model is fairly high. We can try some hyperparameter tuning to improve it.

# In[70]:


## Plot the ROC curve for our model
plot_roc_curve(logreg_rfe_baseline, X_test_rfe, y_test, drop_intermediate=True);


# ### 📌 We get a good ROC curve with Area under curve (AUC) = 0.9

# # <font color=steelblue>🔧 Hyperparameter Tuning</fonts>

# In[71]:


## Define a parameter dictionary
params = {
    'penalty': ['l1', 'l2','elasticnet', 'none'],
    'C': [0.001,0.01,0.1,1,10,100],
    'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}


# In[72]:


## Search from 5 folds for the best parameters from dictionary above
grid_search = GridSearchCV(estimator=logreg_rfe_baseline,
                           param_grid=params,
                           cv = 5,
                           n_jobs=-1, verbose=1, scoring="recall")


# In[73]:


## Fit the search on training data
grid_search.fit(X_train_rfe, y_train)


# In[74]:


## View the best recall score
grid_search.best_score_


# In[75]:


## View the best estimator hyperparameter values
grid_search.best_params_


# ### 📌 We see that we can improve our `Recall` score from 0.87 to 0.90 by using the above values of hyperparamters. So, now we build a logistic regresion model using these values of hyperparameters.

# In[76]:


## Instantiate a Logistic Regression Model using tuned hyperparameters
logreg_rfe_tuned = LogisticRegression(penalty='l1',
                                      C=0.001,
                                      class_weight='balanced',
                                      random_state=42, 
                                      solver='liblinear',
                                      n_jobs=-1,
                                      verbose=1)


# In[77]:


## Train the tuned Logistic Regression model with 15 selected features
logreg_rfe_tuned.fit(X_train_rfe, y_train)


# # <font color=steelblue>🎯 Model Evaluation</fonts>

# In[78]:


## Make predictions on the test data
y_pred = logreg_rfe_tuned.predict(X_test_rfe)
y_pred


# In[79]:


## View predicted probabilities
preds = logreg_rfe_tuned.predict_proba(X_test_rfe)
preds


# In[80]:


## View the confusion matrix
print(confusion_matrix(y_test, y_pred))

fig, ax = plt.subplots(figsize=(15,8))
plot_confusion_matrix(logreg_rfe_tuned, X_test_rfe, y_test, display_labels=['Not_Churn', 'Churn'], ax=ax);


# In[81]:


## View the recall score
recall_rfe_tuned = recall_score(y_test, y_pred)
recall_rfe_tuned = round(recall_rfe_tuned, 2)
recall_rfe_tuned


# ### 📌 The Recall score of our logistic regression model has improved from `0.87` to `0.91` by using the tuned hyperparameters

# In[82]:


## Plot the ROC curve for our model
plot_roc_curve(logreg_rfe_tuned, X_test_rfe, y_test, drop_intermediate=True);


# # <font color=steelblue>🔍 Identifying Important Churn Predictors</fonts>

# In[83]:


## Create a dataframe of important features in their descending order of absolute importance in prediction
feature_importance=pd.DataFrame({'feature':list(X_train_rfe.columns), 'feature_importance':[i for i in logreg_rfe_tuned.coef_[0]]  ,'feature_importance_absolute':[abs(i) for i in logreg_rfe_tuned.coef_[0]]})
feature_importance.sort_values('feature_importance_absolute',ascending=False, inplace=True)
feature_importance


# In[84]:


## Visualize the features and their importance
plt.figure(figsize=(10,6))
plt.title("Feature Importances", fontdict={'fontsize':25, 'color':'navy'})
sns.barplot(x='feature', y='feature_importance_absolute', data=feature_importance)
plt.xticks(rotation=90);


# # <font color=steelblue>📊 Business Recommendations</fonts>

# ### 📌 The important predictor attributes which help the business understand indicators of churn are:-
# 
# 1. **`total_ic_mou_8`**: If a customer's total incoming voice call usage drops in the month of August, then the customer is highly likely to churn.
# 
# 
# 2. **`last_day_rch_amt_8`**: If a customer's last recharge amount reduces in August, then the probability of that customer getting churned increases.
# 
# 
# 3. **`days_since_last_rech_8`**: The more the number of days past since last recharge in August, the more the chances of that customer being churned.
# 
# 
# 4. **`loc_og_t2m_mou_8`**: If a customer's local outgoing calls (minutes of usage) in August within same telecom circle as well as to other mobile operator decrease, then chances of churn increase.
# 
# 
# 5. **`vol_3g_mb_8`**: If a customer's 3G internet usage for the month of August decreases, then he or she is likely to churn.
# 
# 
# 6. **`loc_og_t2t_mou_8`**: If a customer's local outgoing calls (minutes of usage) in August within same operator decrease, then chances of churn increase.
# 
# 
# 7. **`std_og_mou_8`**: If a customer's std outgoing calls (minutes of usage) in August decrease, then the chances of churn increase.
# 

# ###  📌 To prevent the churner from leaving their company, the telecom company may adopt one or more of the following strategies:-
# 

# - **`Network Connectivity`** - Checking the network connectivity of the geography and improving if necessary may resolve the issue of less number of incoming voice calls and usage of 3G internet resulting in less churn. Improving call drop issues.
# 
#  
# 
# - **`Offers on Recharge`** - Giving extra talktime and/or additional internet data on same recharge amount.
# 
#  
# 
# - **`Discounts on Recharge amount`** - If the number of days past since last recharge in August are high, those customers will be attracted and might go for recharge from other telecom company. They could be given attractive discounts on recharge to prevent them from churning.
# 
#  
# 
# - **`Doubling the 3G usage limit`** per day for free may reduce the churn and attract new customers.
# 
#  
# 
# - **`Local & STD Outgoing calls`** within the same operator should be made free to reduce churn and increase more business.

# ### 📌 Now we create a summary table for comparing this model and all the models we will build from now onwards

# In[85]:


## Creating a dataframe to keep track of all models built and their Recall scores
summary = pd.DataFrame([{'Model': 'Logistic Regression (with RFE)','Recall Score (Baseline)': recall_rfe_baseline, 'Recall Score (Tuned)': recall_rfe_tuned}])
summary


# # <font color=midnightblue>Step-3:</font>  🚀 <font color=mediumvioletred>Model Building and Evaluation # 2 : High Performance Model</font>

# # <font color=steelblue>💠 Dimensionality Reduction (using PCA)</fonts>

# In[86]:


## Create a PCA object (for retaining components that explain 95% of the variance)
pca95 = PCA(n_components=0.95, random_state=42)

## Fit the data
pca95.fit(X_train)

## Get PCA coordinates for scaled_data
X_train_pca = pca95.transform(X_train) 

## Also transform test data
X_test_pca = pca95.transform(X_test)


# In[87]:


## View the features before and after transformation
print(f"Original number of features: {X_train.shape[1]} \nReduced number of features: {X_train_pca.shape[1]}")


# ### 📌 So, we have reduced our features from 129 to 72. These 72 Principal Components explain 95% of variance in our data.

# In[88]:


## View the explained variance ratio of all 72 components

per_var = np.round(pca95.explained_variance_ratio_* 100, 1)

## Create labels for ease of understanding the Principal Components
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

per_var


# In[89]:


## Plot the Percentage of Explained Variance and Cumulative Percentage of Explained variance using scree plots

fig,ax = plt.subplots(figsize=(15,8))
ax.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance (in %)')
plt.xlabel('Principal Component')
plt.xticks(rotation=90, fontsize=12)
plt.title('Scree Plot')

ax2=ax.twinx()
ax2.set_ylabel("Cumulative Percentage of Explained Variance (in %)")
var_cumu = np.cumsum(pca95.explained_variance_ratio_*100)
ax2.plot(range(1,len(var_cumu)+1), var_cumu, 'g-.' )

plt.show()


# In[90]:


## Summary table of variance explained
pca_summary = pd.DataFrame({'Principal Components':labels,
                        'Variable Explained (in %)':per_var,
                        'Cumulative Variance Explained (in %)':np.round((var_cumu),1)})
pca_summary


# ### 📌 From the above plots and summary table, we observe that about 95% of the variance is explained by the first 72 components.

# In[91]:


## Converting X_train_pca to a dataframe with suitable column names
X_train_pca = pd.DataFrame(X_train_pca, columns=[f"PC{i}" for i in range(1,73)])
X_train_pca.head()


# In[92]:


## Rebuilding our dataset by concatenating target column (it will be useful in some visualizations below)
telecom_pca = pd.concat([X_train_pca, y], axis=1)
telecom_pca.head(10)


# # 📊 Plotting the data in terms of first two Principal Components

# In[93]:


## View the data points in a scatterplot of PC1 v/s PC2
sns.pairplot(data=telecom_pca, x_vars=["PC1"], y_vars=["PC2"], hue = "churn", size=8);


# ### 📌 We can see here that PC1 and PC2 capture much variance of data

# In[94]:


## Check the multicollinearity
plt.figure(figsize=(15,8))
sns.heatmap(telecom_pca.corr(), cmap='viridis');


# ### 📌 So, the selected principal components have zero correlations among them and with target variable `churn`. We see that PCA handles multicollinearity very efficiently.

# ## Now, we will build three different models here using the 72 Principal Components
# ### 1. Logistic Regression
# ### 2. Random Forest
# ### 3. XGBoost Classifier

# In[95]:


## Create helper function to evaluate the model on test data
def evaluate_model(dt_classifier):
    ''' This functions takes input as a classifier, prints the Recall score and plots the Confusion Matrix
        This function also returns the Recall Score of classifier on test data, rounded by two decimal points'''
    recall = round(recall_score(y_test, dt_classifier.predict(X_test_pca)), 2)
    print("Recall :", recall_score(y_test, dt_classifier.predict(X_test_pca)))
    print("Test Confusion Matrix:")
    fig, ax = plt.subplots(figsize=(15,8))
    print(plot_confusion_matrix(dt_classifier, X_test_pca, y_test, display_labels=['Not_Churn', 'Churn'], ax=ax))
    plt.show()
    return recall


# # <font color=steelblue>(1) Logistic Regression  0️⃣📈1️⃣</font>
# ## <font color=steelblue> (Model Building, Hyperparameter Tuning and Model Evaluation)</font>

# In[96]:


## Build a baseline logistic regression model with only features selected by PCA
###################################################################################
## To HANDLE CLASS IMBALANCE in data, we set 'class_weight' parameter to 'balanced'
###################################################################################
logreg_pca_baseline = LogisticRegression(random_state=42, n_jobs=-1, class_weight='balanced')


# In[97]:


## Train the baseline Logistic Regression model with 72 features selected by PCA
logreg_pca_baseline.fit(X_train_pca, y_train)


# In[98]:


## Evaluate the baseline logistic regression model on test data
recall_pca_baseline = evaluate_model(logreg_pca_baseline)


# In[99]:


## View the recall score
recall_pca_baseline


# ### 📌 The Recall score we are getting in this baseline model is fairly high. 

# ### 📌 Now, let's try to see if we can improve the recall score using hyperparameter tuning

# In[100]:


## Define a parameter dictionary
params = {
    'penalty': ['l1', 'l2','elasticnet', 'none'],
    'C': [0.001,0.01,0.1,1,10,100],
    'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}


# In[101]:


## Search from 5 folds for the best parameters from dictionary above
grid_search = GridSearchCV(estimator=logreg_pca_baseline,
                           param_grid=params,
                           cv = 5,
                           n_jobs=-1, verbose=1, scoring="recall")


# In[102]:


## Fit the gridsearch on training data (PCA)
grid_search.fit(X_train_pca, y_train)


# In[103]:


## View the best recall score 
grid_search.best_score_


# In[104]:


## View the best hyperparameter values
grid_search.best_params_


# ### 📌 So, we can improve our `Recall` score from 0.85 to 0.88 by using the above values of hyperparamters. 
# ### 📌 Now we build a logistic regresion model using these values of hyperparameters.

# In[105]:


## Instantiate a Logistic Regression Model using tuned hyperparameters
logreg_pca_tuned = LogisticRegression(C=0.001,
                                      penalty='l2',
                                      class_weight='balanced',
                                      random_state=42, 
                                      solver='liblinear',
                                      n_jobs=-1,
                                      verbose=1)


# In[106]:


## Train the Logistic Regression model with features selected by PCA
logreg_pca_tuned.fit(X_train_pca, y_train)


# In[107]:


## Evaluate the tuned model on test data
recall_pca_tuned = evaluate_model(logreg_pca_tuned)


# In[108]:


## View the recall score
recall_pca_tuned


# ### 📌 The Recall score of our logistic regression model has slightly improved from `0.85` to `0.87` by using the tuned hyperparameters

# In[109]:


## Plot the ROC curve for our model
plot_roc_curve(logreg_pca_tuned, X_test_pca, y_test, drop_intermediate=True);


# In[110]:


## Adding this model to our summary table
summary.loc[len(summary.index)] = ['Logistic Regression (with PCA)', recall_pca_baseline, recall_pca_tuned]
summary


# # <font color=steelblue>(2) Random Forest 🌳🌲</font>
# ## <font color=steelblue> (Model Building, Hyperparameter Tuning and Model Evaluation)</font>

# In[111]:


## Build a baseline Random Forest with only features selected by PCA
##################################################################################
## To HANDLE CLASS IMBALANCE in data, we set 'class_weight' parameter to 'balanced'
##################################################################################
rf_baseline = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')


# In[112]:


## Train the Random Forest model with 72 selected features
rf_baseline.fit(X_train_pca, y_train)


# In[113]:


## Evaluate the baseline Random Forest model on test data
recall_rf_baseline = evaluate_model(rf_baseline)


# In[114]:


## View the recall score
recall_rf_baseline


# ### 📌 The Recall score we are getting in baseline Random Forest model is very low. 

# ### 📌 Now, let's try to see if we can improve the recall score using hyperparameter tuning

# In[115]:


# Create the parameter grid based on the results of random search 

params =  {'max_depth': [1, 2, 5, 10, 20],
            'min_samples_leaf': [50, 100, 200, 300],
            'max_features': [4,5, 6, 7],
            'n_estimators': [500, 600, 700, 800],
            'criterion':['gini','entropy']
          }


# In[116]:


## Instantiate the grid search model
## Here, we have used RandomizedSearchCV to speed up the search process without affecting the performance too much
grid_search = RandomizedSearchCV(estimator=rf_baseline, 
                                 param_distributions=params, 
                                 cv=5, 
                                 n_jobs=-1, 
                                 verbose=1, 
                                 scoring = "recall")


# In[117]:


## Fit the randomized search on training data
grid_search.fit(X_train_pca,y_train)


# In[118]:


## View the best recall score 
grid_search.best_score_


# In[119]:


## View the best hyperparameter values
grid_search.best_params_


# ### 📌 So, we can improve our `Recall` score from 0.19 to 0.77 by using the above values of hyperparamters. 
# ### 📌 Now we build a Random Forest model using these values of hyperparameters.

# In[120]:


## Instantiate a Random Forest Model using tuned hyperparameters
rf_tuned = RandomForestClassifier(n_estimators=800,
                                  min_samples_leaf=200,
                                  max_features=6,
                                  max_depth=5,
                                  criterion='gini',
                                  class_weight='balanced',
                                  random_state=42, 
                                  n_jobs=-1,
                                  verbose=1)


# In[121]:


## Train the tuned Random Forest model with features by PCA
rf_tuned.fit(X_train_pca, y_train)


# In[122]:


## Evaluate the tuned model on test data
recall_rf_tuned = evaluate_model(rf_tuned)


# In[123]:


## View the recall score
recall_rf_tuned


# ### 📌 The Recall score of our Random Forest model has improved from `0.19` to `0.78` by using the tuned hyperparameters

# In[124]:


## Adding this model to our summary table
summary.loc[len(summary.index)] = ['Random Forest (with PCA)', recall_rf_baseline, recall_rf_tuned]
summary


# # <font color=steelblue>(3) XG Boost 🚗💨</font>
# ## <font color=steelblue> (Model Building, Hyperparameter Tuning and Model Evaluation)</font>

# ### 📌 As XGBoost algorithm is robust to multicollinearity, we will take our orginal `X_train` and `y_train` to train the model (instead of X_train_pca and y_train_pca)

# In[125]:


## Build a baseline XGBoost Classifier
xgb_baseline = xgb.XGBClassifier()

## Train the XGBoost model on the training data
xgb_baseline.fit(X_train, y_train)


# In[126]:


## Make predictions of baseline XGBoost model on test data
y_pred = xgb_baseline.predict(X_test)

## Round preicted values
predictions = [round(value) for value in y_pred]
predictions[:10]


# In[127]:


## Evaluate the baseline XGBoost model on test data
## View the confusion matrix
print(confusion_matrix(y_test, y_pred))

fig, ax = plt.subplots(figsize=(15,8))
plot_confusion_matrix(xgb_baseline, X_test, y_test, display_labels=['Not_Churn', 'Churn'], ax=ax);


# In[128]:


## View the recall score 
recall_xgb_baseline = round(recall_score(y_test, predictions), 2)
recall_xgb_baseline


# ### 📌 The Recall score we are getting in baseline Random Forest model is low. 

# ### 📌 Now, let's try to see if we can improve the recall score using hyperparameter tuning

# In[129]:


### First we select a suitable value of 'scale_pos_weight' parameter in order to HANDLE CLASS IMBALANCE
y_train.value_counts()


# In[130]:


## `scale_pos_weight` is the ratio of number of negative class to the positive class
scale_pos_weight = y_train.value_counts().loc[0]/y_train.value_counts().loc[1]
scale_pos_weight


# In[131]:


# Create the parameter grid based on the results of random search 
parameters = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
              'max_depth': [2, 4, 6, 8, 10],
              'min_child_weight': [3, 7, 11, 19, 25],
              'scale_pos_weight': [10.57],
              'n_estimators': [50, 100, 150, 200, 300, 500]}


# In[132]:


## Instantiate the grid search model
## Here, we have used RandomizedSearchCV to speed up the search process without affecting the performance too much
grid_search = RandomizedSearchCV(estimator=xgb_baseline,
                                 param_distributions=parameters,
                                 n_jobs=-1,
                                 cv=5,
                                 scoring='recall',
                                 verbose=1)

grid_search.fit(X_train, y_train)


# In[133]:


## View the recall score
print(grid_search.best_score_)


# ### 📌 The Recall score of our XGBoost model has slightly improved from 0.55 to 0.84 by using the tuned hyperparameters. Now, we build a XGBoost model using these best hyperparameters

# In[134]:


## View the best estimator hyperparameters
grid_search.best_params_


# In[135]:


## Instantiate a XGBoost Model using tuned hyperparameters
xgb_tuned = xgb.XGBClassifier(scale_pos_weight=10.57,
                              n_estimators=100,
                              min_child_weight=19,
                              max_depth=2,
                              learning_rate=0.3,
                              random_state=42, 
                              n_jobs=-1,
                              verbose=1)


# In[136]:


## Train the XCBoost classifier model with training data
xgb_tuned.fit(X_train, y_train)


# In[137]:


## Make predictions of tuned XGBoost model on test data
y_pred = xgb_tuned.predict(X_test)

## Round preicted values
predictions = [round(value) for value in y_pred]
predictions[:10]


# In[138]:


## Evaluate the tuned XGBoost model on test data
## View the confusion matrix
print(confusion_matrix(y_test, y_pred))

fig, ax = plt.subplots(figsize=(15,8))
plot_confusion_matrix(xgb_tuned, X_test, y_test, display_labels=['Not_Churn', 'Churn'], ax=ax);


# In[139]:


## View the recall score 
recall_xgb_tuned= round(recall_score(y_test, predictions), 2)
recall_xgb_tuned


# ### 📌 The Recall score of our Random Forest model has improved from `0.55` to `0.85` by using the tuned hyperparameters

# In[140]:


## Adding this model to our summary table
summary.loc[len(summary.index)] = ['XGBoost', recall_xgb_baseline, recall_xgb_tuned]
summary


# # <font color=limegreen> 📝 Summarizing all the models created and selecting the best one</font>

# In[141]:


## Summary table
summary.set_index('Model', inplace=True)
summary


# ### 📌 Above table shows that if we use a simple Logistic Regression model with features selected by RFE technique, we get a decent recall score of 0.91.
# 
# 
# ### 📌 The other three models: Logistic Regression (with PCA), Random Forest (with PCA) and XGBoost also give good recall scores, but they take up a lot of processing power. Also, there is a scope of doing more intense hyperparameter tuning which may improve the recall score.
# 
# 
# ### 📌 If the telecom company wants to improve `churn` prediction and can afford high processing power, then we should go ahead with any of the latter three models and do some more hyperparameter tuning to improve the recall score even further.
# 
# 
# ## ✔️ However, if the telecom company wants to pinpoint the features to focus on in order to prevent `churn`, then the first model: <font color='brown'>Logistic Regression (with RFE)</font> is the one that should be implemented.
