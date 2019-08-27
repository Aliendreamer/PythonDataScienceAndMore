#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

#%%
data = pd.read_csv(r'DataScienceBootcampCourse\csvFiles\real_estate_price_size.csv')

#%%
data.head()
data.describe()

#%%
x = data['size']
y = data['price']

plt.scatter(x,y)
plt.xlabel('Size',fontsize=20)
plt.ylabel('Price',fontsize=20)
plt.show()


#%%
x_matrix = x.values.reshape(-1,1)

reg = LinearRegression()
reg.fit(x_matrix,y)

#%%
reg.score(x_matrix,y)

#%%
reg.coef_


#%%
x =np.array([750]).reshape(-1,1);
reg.predict(x)

#%%


#%%
