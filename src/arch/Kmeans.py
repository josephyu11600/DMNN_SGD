import numpy as np
from scipy.cluster.vq import kmeans, vq

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

#%%



def find_optimal_clusters(data, max_k=10): #EDIT kmax

    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.show()
    
    optimal_k = np.argmin(inertias) + 1 #EDIT  Tolerance score
    return optimal_k

def calculate_centroids_and_radii(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    radii = []
    for i in range(n_clusters):
        cluster_points = data[kmeans.labels_ == i]
        distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
        radii.append(distances.max())
    
    return centroids, radii



#%% Local Test

if __name__ == "__main__":

    data = np.random.rand(8300,23*256)
    optimal_k = find_optimal_clusters(data, max_k=100)
    centroids, radii = calculate_centroids_and_radii(data, optimal_k)

    print("Centroids:", centroids)
    print("Radii:", radii)


# %%
