#%%
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#%%
wineData=pd.read_csv(r"./data/winequality-red.csv",sep=";");
print(wineData)


#%%
wineData.shape

#%%
wine_features = wineData.drop("quality", axis = 1)

#%%
regression=LinearRegression()
regression.fit(wine_features,wineData.quality)

#%%
regression.score(wine_features,wineData.quality)

#%%
regression.intercept_ 

#%%
polynomialWineSet = PolynomialFeatures(2)

#%%
wineSet =polynomialWineSet.fit_transform(wine_features)

#%%
wineSet.shape

#%%
wineSet.describe

#%%
regression.fit(wineSet,wineData.quality)

#%%
regression.score(wineSet,wineData.quality)

#%%
