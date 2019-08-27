# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'DataScienceBootcampCourse\\notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Adjusted R-squared - Exercise
# 
# Using the code from the lecture, create a function which will calculate the adjusted R-squared for you, given the independent variable(s) (x) and the dependent variable (y).
# 
# Check if you function is working properly.
# 
# Please solve the exercise at the bottom of the notebook (in order to check if it is working you must run all previous cells).
#%% [markdown]
# ## Import the relevant libraries

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

#%% [markdown]
# ## Load the data

#%%
data = pd.read_csv(r'DataScienceBootcampCourse/csvFiles/1.02.Multiple-linear-regression.csv')
data.head()


#%%
data.describe()

#%% [markdown]
# ## Create the multiple linear regression
#%% [markdown]
# ### Declare the dependent and independent variables

#%%
x = data[['SAT','Rand 1,2,3']]
y = data['GPA']

#%% [markdown]
# ### Regression itself

#%%
reg = LinearRegression()
reg.fit(x,y)


#%%
reg.coef_


#%%
reg.intercept_

#%% [markdown]
# ### Calculating the R-squared

#%%
reg.score(x,y)

#%% [markdown]
# ### Formula for Adjusted R^2
# 
# $R^2_{adj.} = 1 - (1-R^2)*\frac{n-1}{n-p-1}$

#%%
x.shape


#%%
r2 = reg.score(x,y)
n = x.shape[0]
p = x.shape[1]

adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
adjusted_r2

#%% [markdown]
# ### Adjusted R^2 function

#%%



#%%



