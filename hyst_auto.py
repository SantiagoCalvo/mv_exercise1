#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Automatic hysteresis thresholding

Author: FILL IN
MatrNr: FILL IN
"""

import cv2
import numpy as np

from hyst_thresh import hyst_thresh


def hyst_thresh_auto(edges_in: np.array, low_prop: float, high_prop: float) -> np.array:
    """ Apply automatic hysteresis thresholding.

    Apply automatic hysteresis thresholding by automatically choosing the high and low thresholds of standard
    hysteresis threshold. low_prop is the proportion of edge pixels which are above the low threshold and high_prop is
    the proportion of edgepixels above the high threshold.

    :param edges_in: Edge strength of the image in range [0., 1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low_prop: Proportion of pixels which should lie above the low threshold
    :type low_prop: float in range [0., 1.]

    :param high_prop: Proportion of pixels which should lie above the high threshold
    :type high_prop: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################
    # Write your own code here
    edges_in_copy = edges_in.copy()
    edges_pixels = edges_in_copy[edges_in > 0]

    low_threshold = np.percentile(edges_pixels, 100*(1-low_prop))
    high_threshold = np.percentile(edges_pixels, 100*(1-high_prop))

    print(low_threshold)
    print(high_threshold)

    hyst_out = hyst_thresh(edges_in=edges_in, low=low_threshold, high=high_threshold)
    ######################################################
    return hyst_out
