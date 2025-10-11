#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Blur the input image with Gaussian filter kernel

Author: FILL IN
MatrNr: FILL IN
"""

import cv2
import numpy as np

def blur_gauss(img: np.array, sigma: float) -> np.array:
    """ Blur the input image with a Gaussian filter with standard deviation of sigma.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma: The standard deviation of the Gaussian kernel
    :type sigma: float

    :return: Blurred image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0.,1.]
    """
    ######################################################
    # Write your own code here
    kernel_width = int(2 * np.ceil(3 * sigma) + 1)
    kernel_space = np.linspace(-(kernel_width)/2., (kernel_width)/2., kernel_width)
    xx, yy = np.meshgrid(kernel_space, kernel_space)
    
    gaussian_kernel = np.exp(-((xx**2 + yy**2)/(2*sigma**2))) / (2*np.pi*sigma**2)
    img_blur = cv2.filter2D(src=img, ddepth=-1, kernel=gaussian_kernel)

    ######################################################
    return img_blur
