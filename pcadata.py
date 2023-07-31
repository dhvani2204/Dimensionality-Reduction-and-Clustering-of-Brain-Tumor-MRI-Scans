import numpy
import numpy as np
from sklearn.decomposition import PCA
import data

CANCERS = {1.0: "Meningioma", 2.0: "Glioma", 3.0: "Pituitary"}

for cancer in CANCERS.keys():
    SAMPLES = 240
    images = data.load_data(SAMPLES, cancer)
    PCAimg = []

    for img in images:
        pca = PCA(n_components=65)
        img_pca = pca.fit_transform(img)
        img_inv = pca.inverse_transform(img_pca)
        PCAimg.append(img_inv)

    X_pca = np.asarray(PCAimg)

    numpy.save(CANCERS[cancer]+'PCAData.npy', X_pca)
