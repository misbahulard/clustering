import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

# Fungsi untuk menghitung jarak euclidean
def euclidean_distance(data, centroids):
    row, col = np.shape(data)
    c_row, c_col = np.shape(centroids)
    distances = np.zeros((row, c_row))

    for x in range(row):
        dist_arr = []
        for c in centroids:
            dist = 0
            for i in range(len(c)):
                dist += pow((data[x][i] - c[i]), 2)
            dist_arr.append(np.sqrt(dist))
        distances[x] = dist_arr

    return distances

# Fungsi mendapatkan rasio eror
# kembalikan rasio eror terkecil
def error_ratio(clusters, labels):
    max_label = np.max(np.unique(labels))
    num_of_labels = len(np.unique(labels))
    error = []
    
    # coba semua kemungkinan pada label cluster
    # dengan cara mengganti 0 ke 1, 1 ke 2
    for l in range(num_of_labels):
        miss = 0
        if l > 0:
            clusters += 1
            clusters[clusters>max_label] = 0

        for x in range(len(clusters)):
            if clusters[x] != labels[x]:
                miss += 1

        error.append((miss / len(labels)) * 100)

    return np.min(error)

# Fungsi untuk menghitung variance
def cluster_variance(data, clusters, k):
    n = []
    cv = []
    var = []
    vw = 0
    vb = 0

    # kelompokkan data tiap cluster
    for x in range(k):
        n.append([data[i] for i in range(len(data)) if clusters[i] == x])
    
    # hitung variance tiap cluster
    for x in range(k):
        d = n[x]    # data pada cluster x
        d_mean = np.mean(n[x], axis=0)  # rata-rata dari data pada suatu cluster
        sum_of_d = 0    # sigma perhitungan data variance

        for di in d:
            res = np.matrix(di - d_mean)
            res = res * res.T   # pangkat dari matriks / vektor
            sum_of_d += res
        
        # simpan variance tiap cluster
        var.append((1/(len(n[x]) - 1)) * sum_of_d)

    print("VAR")
    print(var)

    # within variance
    sum_of_vw = 0   # sigma perhitungan data variance within
    for x in range(k):
        num_of_data_i = len(d[x])
        sum_of_vw += (num_of_data_i - 1 * var[x])

    # simpan variance within (vw)
    vw = (1/(len(data) - 1)) * sum_of_vw
    
    print("VW")
    print(vw.item())

    # between variance
    sum_of_vb = 0
    for x in range(k):
        num_of_data_i = len(d[x])   # jumlah data tiap cluster
        d_mean = np.mean(n[x], axis=0)  # rata-rata dari data tiap cluster
        grand_mean = np.mean(data, axis=0)  # rata-rata dari seluruh data cluster

        mean_sub_mean = np.matrix(d_mean - grand_mean)
        mean_sub_mean = mean_sub_mean * mean_sub_mean.T
        sum_of_vb += (num_of_data_i * mean_sub_mean) 
    
    # simpan variance between (vb)
    vb = (1/(k - 1)) * sum_of_vb

    print("VB")
    print(vb.item())

    # Hitung total variance
    v = vw / vb

    return v.item()

# fungsi untuk generate centroid
def generate_centroid(data, n_clusters):
    row, col = np.shape(data)
    centroids = np.zeros((n_clusters, col))
    for i in range(n_clusters):
        for j in range(col):
            centroids[i][j] = np.random.uniform(np.min(data[:, j]), np.max(data[:, j]))

    return centroids

def main():
    # load dataset
    df = pd.read_csv("iris.csv")
    df['species'] = df['species'].astype('category')
    data = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    labels = df['species'].cat.codes.values
    
    # jumlah k
    k = 3
    C = generate_centroid(data, k)

    print("Generated centroids")
    print("=========================")
    print(C, "\n")
    
    # inisialisasi clusters dan centroid lama
    old_C = np.zeros(C.shape)
    clusters = np.zeros(len(data))

    # lakukan perulangan untuk mengkomputasi K-Means
    while True:
        # hitung jarak data dengan centroid
        D = euclidean_distance(data, C)
        # dapatkan tiap cluster data
        clusters = np.argmin(D, axis=1)

        print("Clusters")
        print("=========================")
        print(clusters, "\n")

        # salin centroid ke centroid lama
        old_C = deepcopy(C)

        # hitung ulang centroid baru dengan menghitung rata-rata dari data tiap cluster
        for i in range(k):
            points = [data[j] for j in range(len(data)) if clusters[j] == i]
            # jika ada cluster yang kosong maka lanjutkan proses karena menghasilkan NaN jika diteruskan
            if len(points) == 0:
                continue
            C[i] = np.mean(points, axis=0)

        print("New Centroid")
        print("=========================")
        print(C, "\n")
        
        # jika sudah konvergen maka hentikan perulangan
        if np.array_equal(C, old_C):
            break
    
    # tampilkan label
    print("Labels")
    print("=========================")
    print(labels, "\n")

    # Hitung Error Ratio
    error = error_ratio(clusters, labels)
    print("\n\nERROR RATIO: ", error, "%") 
    
    # hitung variance
    variance = cluster_variance(data, clusters, k)
    print("\n\nVARIANCE: ", variance)

    # tampilkan menggunakan plot
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], c=clusters, alpha=0.5, s=100)
    plt.scatter(C[:, 0], C[:, 1], c="red", marker="x")
    plt.subplot(1, 2, 2)
    plt.scatter(data[:, 2], data[:, 3], c=clusters, alpha=0.5, s=100)
    plt.scatter(C[:, 2], C[:, 3], c="red", marker="x")
    plt.show()


if __name__ == "__main__":
    main()

