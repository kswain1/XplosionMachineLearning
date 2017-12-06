import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Read in Hip Speed Data and Exit Velocity
michael_df = pd.read_csv('MachineLearningMichaelPittman.csv')

samples = michael_df[['Hip Speed','Exit Velocity']].dropna()

samples = samples.values.reshape(-2,2)

print(samples)

print(michael_df.head())
#set the number of clusters
model = KMeans(n_clusters=3)

#Fit model to points
model.fit(samples)

#assign the labels to each data point
labels = model.predict(samples)

print(labels)

centroids = model.cluster_centers_



x_axis = samples[:,0]
y_axis = samples[:,1]

plt.scatter(x_axis, y_axis, c=labels)
plt.show()
bam = 1


##for x, y, company in zip(xs, ys, companies):   placing the names of the players associate with their dot
    #plt.annotate(company, (x, y), fontsize=5, alpha=0.75)

