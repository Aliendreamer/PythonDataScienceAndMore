# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'DataScienceBootcampCourse'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Multiple Linear Regression with sklearn - Exercise Solution
#%% [markdown]
# You are given a real estate dataset. 
# 
# Real estate is one of those examples that every regression course goes through as it is extremely easy to understand and there is a (almost always) certain causal relationship to be found.
# 
# The data is located in the file: 'real_estate_price_size_year.csv'. 
# 
# You are expected to create a multiple linear regression (similar to the one in the lecture), using the new data. 
# 
# Apart from that, please:
# -  Display the intercept and coefficient(s)
# -  Find the R-squared and Adjusted R-squared
# -  Compare the R-squared and the Adjusted R-squared
# -  Compare the R-squared of this regression and the simple linear regression where only 'size' was used
# -  Using the model make a prediction about an apartment with size 750 sq.ft. from 2009
# -  Find the univariate (or multivariate if you wish - see the article) p-values of the two variables. What can you say about them?
# -  Create a summary table with your findings
# 
# In this exercise, the dependent variable is 'price', while the independent variables are 'size' and 'year'.
# 
# Good luck!
#%% [markdown]
# ## Import the relevant libraries

#%%


#%% [markdown]
# ## Load the data

#%%



#%%


#%% [markdown]
# ## Create the regression
#%% [markdown]
# ### Declare the dependent and the independent variables

#%%


#%% [markdown]
# ### Regression

#%%


#%% [markdown]
# ### Find the intercept

#%%


#%% [markdown]
# ### Find the coefficients

#%%


#%% [markdown]
# ### Calculate the R-squared

#%%


#%% [markdown]
# ### Calculate the Adjusted R-squared

#%%



#%%


#%% [markdown]
# ### Compare the R-squared and the Adjusted R-squared
#%% [markdown]
# Answer...
#%% [markdown]
# ### Compare the Adjusted R-squared with the R-squared of the simple linear regression
#%% [markdown]
# Answer...
#%% [markdown]
# ### Making predictions
# 
# Find the predicted price of an apartment that has a size of 750 sq.ft. from 2009.

#%%


#%% [markdown]
# ### Calculate the univariate p-values of the variables

#%%



#%%



#%%



#%%


#%% [markdown]
# ### Create a summary table with your findings

#%%


#%% [markdown]
# Answer...

