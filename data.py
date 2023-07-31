from PIL import Image
import pandas
import numpy
import cv2
import matplotlib.pyplot as plt


def load_data(n_samples, can):
    if n_samples > 700:
        n_samples = 700

    info = pandas.read_csv('./info_data.csv')
    info = info[info['shape'].str.contains(r"(512, 512)")]

    def imgArr(files):
        images = list()
        for file in files:
            im = Image.open(file).convert('L')
            arr = numpy.asarray(im)
            images.append(arr)
        return images

    meningioma = (numpy.array(imgArr(list(info[info['label'] == 1.0]['fileLocation'].head(n_samples)))) - 127.5) / 127.5
    glioma = (numpy.array(imgArr(list(info[info['label'] == 2.0]['fileLocation'].head(n_samples)))) - 127.5) / 127.5
    pituitary = (numpy.array(imgArr(list(info[info['label'] == 3.0]['fileLocation'].head(n_samples)))) - 127.5) / 127.5
    images = numpy.concatenate([meningioma, glioma, pituitary])

    # images = (numpy.array(imgArr(list(info[info['label'] == can]['fileLocation'].head(n_samples)))) - 127.5) / 127.5

    return images


if __name__ == "__main__":
    imgs = load_data(100, 1.0).reshape(300, -1)

    df = pandas.DataFrame(imgs, columns=list(range(0, 512*512)))
    df.dropna(how='any', inplace=True, axis=1)
    df.to_csv('images.csv')

