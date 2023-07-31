import numpy
import matplotlib.pyplot as plt
import cv2

CANCERS = {1.0: "Meningioma", 2.0: "Glioma", 3.0: "Pituitary"}

for cancer in CANCERS.keys():
    X_pca = numpy.load(CANCERS[cancer]+'PCAData.npy')
    bin_img = []

    for gray_img in X_pca:
        # Scale the pixel values to range from 0 to 255
        scaled_img = numpy.interp(gray_img, (-1, 1), (0, 255))

        # Convert the scaled image to 8-bit unsigned integer format
        uint8_img = numpy.uint8(scaled_img)

        # Threshold the image to get a binary mask
        threshold_value = 75
        max_value = 255
        _, binary_mask = cv2.threshold(uint8_img, threshold_value, max_value, cv2.THRESH_BINARY)

        # Scale the binary mask back to range from -1 to 1
        scaled_binary_mask = numpy.interp(binary_mask, (0, 255), (-1, 1))

        bin_img.append(scaled_binary_mask)

    bin_img = numpy.asarray(bin_img)
    numpy.save(CANCERS[cancer] + 'BINData.npy', bin_img)