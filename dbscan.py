import numpy
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


X_pca = numpy.load('PCAData.npy')

for data in X_pca:
    clustering = DBSCAN(min_samples=24).fit(data)
    labels = clustering.labels_

    X_pca = PCA(n_components=2).fit_transform(data)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')

    plt.title('DBCSCAN clustering of Brain Tumor Images')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('dbscan.jpg')
    break
