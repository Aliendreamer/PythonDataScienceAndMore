#%% 
import seaborn as sns
import scipy
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
sns.set()
#%%
data=pd.read_csv(r'DataScienceBootcampCourse\csvFiles\real_estate_price_size.csv')
print(data)

#%%
data.describe()


#%%
y=data['size']
x1=data['price']
plt.scatter(x1,y)
plt.xlabel("price",fontsize=20)
plt.ylabel('size',fontsize=20)
plt.show()

#%%
x=sm.add_constant(x1)
#%%
results=sm.OLS(y,x).fit()

results.summary()

#%%
plt.scatter(x1,y)
yhat=0.0033*x1+(-122.3349)
fig=plt.plot(x1,yhat,lw=4,c='red',label='line')
plt.xlabel("price",fontsize=20)
plt.ylabel('size',fontsize=20)
plt.show()


#%%
