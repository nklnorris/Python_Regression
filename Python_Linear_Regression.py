#!/usr/bin/env python
# coding: utf-8

# # Python Linear Regression

# ## Load and investigate WHO Life Expectancy data

# In[199]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Read the CSV file
life = pd.read_csv(r'C:\Users\nklpu\OneDrive\Desktop\Data Vis\ML Exercises\life_expectancy_data.csv')
# Prevew laded data
life.head(n=10)
list(life.columns)


# In[200]:


# Only keep the life expectancy and infant death columns
life2 = life[['Life_expectancy ', 'percentage_expenditure']]
# Verify the desired columns are present
life2.head(n=10)


# In[201]:


# Check number of obervations
print(life2.info())


# In[202]:


# Check for nulls. We see in the dataframe into there are 10 nulls in Life Expectancy
life2.isnull().values.any()
life2.isnull().sum().sum()


# In[203]:


# Drop Life Expectancy rows containing nulls
life3 = life2.copy()
life3['Life_expectancy '].replace('', np.nan, inplace=True)
life3.dropna(subset=['Life_expectancy '], inplace=True)
life3.isnull().sum().sum()


# In[204]:


# Verify nulls were removed
print(life3.info())


# In[205]:


# Create a histogram to visualize the spread of Life Expectancy (check for outliers)
life3['Life_expectancy '].hist()


# In[206]:


# Create a histogram to visualize the spread of BMI (check for outliers)
life3[ 'percentage_expenditure'].hist()


# In[207]:


# There appear to be quite a few 0s in the Percentage Expenditure column. To built an accurate model these should be removed
life3 = life3[life3['percentage_expenditure'] != 0]

# Verify all 0s were removed
print(life3.info())


# In[208]:


# Create a scatterplot showing the relationship between infant deaths and life expectancy
plt.scatter(life3['percentage_expenditure'], life3['Life_expectancy '])
plt.title("The Effect of Health Expenditure on Life Expectancy")
plt.xlabel("Expenditure on health as a percentage of Gross Domestic Product per capita(%)")
plt.ylabel("Life Expectancy")


# In[209]:


# Create scatterplot using log of x variable to normalize the data
plt.scatter(np.log(life3['percentage_expenditure']), life3['Life_expectancy '])
plt.title("The Effect of Health Expenditure on Life Expectancy")
plt.xlabel("Log of Expenditure on health as a % of Gross Domestic Product per capita")
plt.ylabel("Life Expectancy")


# ## Split data into test and training set

# In[210]:


X_train, X_test, Y_train, Y_test = train_test_split(life3['percentage_expenditure'], life3['Life_expectancy '],
                                                    test_size=0.3,
                                                    train_size=0.7,
                                                    random_state=42)


# ## Create model using training set

# In[211]:


# Log transform X variable in training and test data
logX_train = np.log(X_train)
logX_test = np.log(X_test)

# Transforming x variables for reression
logX_train = logX_train.values.reshape(-1, 1)
logX_test = logX_test.values.reshape(-1, 1)

# Create the linear regression model using the log transformed x variable
model = LinearRegression().fit(logX_train, Y_train)


# ## Graph model onto scatterplot

# In[212]:


plt.scatter(np.log(life3['percentage_expenditure']), life3['Life_expectancy '])
plt.title("The Effect of Health Expenditure on Life Expectancy")
plt.xlabel("Log of Expenditure on health as a % of Gross Domestic Product per capita")
plt.ylabel("Life Expectancy")
plt.plot(logX_train, model.predict(logX_train), color = 'red')


# ## Use model to predict life expectancy of test data set and evaluate algorithm's fit

# In[213]:


# Get mean absollute error on training data to use in comparison to the test data
Y_train_pred = model.predict(logX_train)
print("Mean Absolute Error on training data = " , metrics.mean_absolute_error(Y_train, Y_train_pred))


# In[214]:


# Get mean absolute error on test data
Y_test_pred = model.predict(logX_test)
print("Mean Absolute Error on test data = " , metrics.mean_absolute_error(Y_test, Y_test_pred))


# In[215]:


# Calculate average life expectancy from full data set
sum((life3['Life_expectancy '])) / len((life3['Life_expectancy ']))


# ### The MAE is between 5.5 and 6, meaning the average error between predicted life expectancy and actual life expectancy is between 5.5 and 6 years. For reference the average life expectancy is 69 years.

# In[ ]:




