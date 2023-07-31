import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import numpy
import saveClusters

n_clusters = 9

CANCERS = {1.0: "Meningioma", 2.0: "Glioma", 3.0: "Pituitary"}

for cancer in CANCERS.keys():
    data = numpy.load(CANCERS[cancer] + 'BINData.npy')
    data = data.reshape(240, -1)

    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='complete').fit(data)
    labels = clustering.labels_
    cluster_counts = [sum(labels == i) for i in range(n_clusters)]

    X_pca = PCA(n_components=2).fit_transform(data)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')

    plt.title('Agglomerative Clustering of Brain Tumor Images')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.savefig(CANCERS[cancer]+'Agglomerative.jpg')
    saveClusters.save(CANCERS[cancer]+'agglomerative', labels, 1.0)
