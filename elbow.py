import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch

# CANCERS = {1.0: "Meningioma", 2.0: "Glioma", 3.0: "Pituitary"}
for x in range(3):

    data = numpy.load('MeningiomaBINData.npy')
    data = data.reshape(240, -1)
    wcss = []

    for i in range(4, 12):

        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    plt.plot(range(4, 12), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig(CANCERS[cancer]+'Elbow.jpg')
    plt.clf()
