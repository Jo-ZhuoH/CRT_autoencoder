# -*- coding: utf-8 -*-"
"""
Created on 12/15/2021  2:22 PM


@author: Zhuo
"""
import cv2
import six
import numpy as np
import skimage.measure
from skimage.feature.texture import greycomatrix, greycoprops

import radiomics
from radiomics import featureextractor, firstorder, glcm, imageoperations, shape, glrlm, glszm


def get_parameters(image, entropy=False, radiomic=False):
    params_values = []
    params_names = []

    if entropy:
        entropy = skimage.measure.shannon_entropy(image)
        params_values.append(entropy)
        params_names.append("original_shannon_entropy")

    if radiomic:
        # mask
        mask = np.ma.masked_greater(image, 0).mask.astype(np.uint8)
        assert mask.sum() > 3000, "Sum is not > 3000, is {}".format(mask.sum())

        cv2.imwrite("test_image.jpg", image)
        cv2.imwrite("test_mask.jpg", mask)

        extractor = featureextractor.RadiomicsFeatureExtractor()
        result = extractor.execute("test_image.jpg", "test_mask.jpg")

        for key, value in six.iteritems(result):
            params_names.append(key)
            if isinstance(value, np.ndarray):
                value = float(value)
            params_values.append(value)
            # print("\t", key, ":", value)

    return params_values, params_names


if __name__ == '__main__':
    img_path = "/home/zhuo/Desktop/CRT_autoencoder/test_image.npy"
    img = np.load(img_path)
    values, names = get_parameters(img, others=True)
    print(values)
    print(names)

