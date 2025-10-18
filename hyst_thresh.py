#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Hysteresis thresholding

Author: Santiago Calvo Salazar
MatrNr: 12450801
"""

import cv2
import numpy as np


def hyst_thresh(edges_in: np.array, low: float, high: float) -> np.array:
    """ Apply hysteresis thresholding.

    Apply hysteresis thresholding to return the edges as a binary image. All connected pixels with value > low are
    considered a valid edge if at least one pixel has a value > high.

    :param edges_in: Edge strength of the image in range [0.,1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low: Value below all edges are filtered out
    :type low: float in range [0., 1.]

    :param high: Value which a connected element has to contain to not be filtered out
    :type high: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################
    # Write your own code here
    weak_and_strong_pixels = np.where(edges_in >= low, edges_in, 0)

    image_int8 = (weak_and_strong_pixels * 255).astype(np.uint8)

    _, labels = cv2.connectedComponents(image_int8, connectivity=8)
    
    strong_pixels_mask = (edges_in > high)

    strong_pixels_labels = np.unique(labels[strong_pixels_mask])

    egde_pixels_mask = np.isin(labels, strong_pixels_labels)

    bitwise_img = egde_pixels_mask * 1.0
    
    ######################################################
    return bitwise_img
