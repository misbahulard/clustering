from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt

def main():
    print("K-Means Clustering")
    df = pd.read_csv("iris.csv")
    data = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)
    y_means = kmeans.predict(data)
    centers = kmeans.cluster_centers_

    plt.scatter(data[:, 0], data[:, 1], c=y_means, alpha=0.8)
    plt.scatter(centers[:, 0], centers[:, 1], c="red")
    plt.show()

if __name__ == "__main__":
    main()
        
