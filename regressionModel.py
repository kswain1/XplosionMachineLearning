import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

player_df = pd.read_csv('MachineLearningKehlin.csv')
print player_df.head()

print(player_df.corr())
#ax = sns.heatmap(data=player_df.corr(), square=True, cmap='RdYlGn')
data = 9

## Features and targets
print(player_df['Distance'].isnull().values.any())
print(player_df['Launch Angle'].isnull().values.any())

X = player_df['Launch Angle'].dropna()
Y = player_df['Distance'].dropna()

Y = Y.values.reshape(-1,1)
X = X.values.reshape(-1,1)

##Machine learning for Regression Modelling
reg = LinearRegression()

#Create prediction space
prediction_space = np.linspace(min(X), max(X)).reshape(-1,1)

#fit the model to the data

reg.fit(X,Y)

#compute predictions over predictions sapce
y_pred = reg.predict(prediction_space)

y_scatter = reg.predict(X)


#Print R^2
print(reg.score(X,Y))

plt.scatter(X,Y)
#plt.plot(prediction_space,y_pred,color='black',linewidth =3)
plt.scatter(X,y_scatter)

plt.show()

data = 0
