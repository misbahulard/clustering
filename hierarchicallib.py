from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from matplotlib import pyplot as plt

def main():
    df = pd.read_csv("iris.csv")
    data = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    hierarchical = AgglomerativeClustering(n_clusters=3)
    hierarchical.fit(data)
    labels = hierarchical.labels_
    #print(hierarchical.labels_)
    plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.8)
    #plt.scatter(centers[:, 0], centers[:, 1], c="red")
    plt.show()

if __name__ == "__main__":
    main()
        
