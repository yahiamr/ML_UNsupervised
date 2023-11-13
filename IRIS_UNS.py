from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris()

iris_df = pd.DataFrame(iris.data,columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['target_name']= iris_df['target'].map(lambda features: iris.target_names[features])



print(iris_df.describe())

sns.pairplot(iris_df, hue='target_name')
plt.show()

features = iris_df.iloc[:,:-2].values


model = KMeans(n_clusters=3)
model.fit(features)

iris_df['cluster'] = model.labels_

pd.crosstab(iris_df['target_name'], iris_df['cluster'])

print(model.labels_)
# Plotting the clusters
plt.scatter(features[model.labels_ == 0, 0], features[model.labels_ == 0, 1], label='Cluster 1')
plt.scatter(features[model.labels_ == 1, 0], features[model.labels_ == 1, 1], label='Cluster 2')
plt.scatter(features[model.labels_ == 2, 0], features[model.labels_ == 2, 1], label='Cluster 3')

# Plotting centroids
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=100, c='red', label='Centroids')

plt.legend()
plt.show()