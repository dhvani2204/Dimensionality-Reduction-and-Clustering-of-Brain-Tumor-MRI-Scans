import os
import data
import numpy
from PIL import Image


def save(folder_name, labels, can):
    current_directory = os.getcwd()
    algo_directory = os.path.join(current_directory, folder_name)

    if not os.path.exists(algo_directory):
        os.makedirs(algo_directory)

    n_clusters = len(numpy.unique(labels))
    for x in range(n_clusters):
        cluster_directory = os.path.join(algo_directory, str(x))
        if not os.path.exists(cluster_directory):
            os.makedirs(cluster_directory)

    images = data.load_data(len(labels), can)

    for index, label in enumerate(labels):
        I = images[index]
        I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(numpy.uint8)
        img = Image.fromarray(I8)
        img.save("{}/{}/{}.png".format(algo_directory, label, index))
