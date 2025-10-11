#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Non-Maxima Suppression

Author: FILL IN
MatrNr: FILL IN
"""

import cv2
import numpy as np


def non_max(gradients: np.array, orientations: np.array) -> np.array:
    """ Apply Non-Maxima Suppression and return an edge image.

    Filter out all the values of the gradients array which are not local maxima.
    The orientations are used to check for larger pixel values in the direction of orientation.

    :param gradients: Edge strength of the image in range [0.,1.]
    :type gradients: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param orientations: angle of gradient in range [-np.pi, np.pi]
    :type orientations: np.array with shape (height, width) with dtype = np.float32 and values in the range [-pi, pi]

    :return: Non-Maxima suppressed gradients
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    """
    ######################################################
    # Write your own code here
    orientations = np.rad2deg(orientations)

    nearest_aproximation = 45
    aprox_orientations = (np.round(orientations/45.)*nearest_aproximation).astype("int")

    print(np.unique(aprox_orientations))
    print(np.unique(orientations))

    # use roll to get the different neighbors of the center pixel in all directions in the center pixel
    # in order to later compare the full matrix
    top = np.roll(gradients, 1, axis=0) # equivalent to i-1, j
    bottom = np.roll(gradients, -1, axis=0)  # equivalent to i+1, j
    left = np.roll(gradients, 1, axis=1) # equivalent to i, j-1
    right = np.roll(gradients, -1, axis=1) # equivalent to i, j+1
    top_left = np.roll(top, 1, axis=1) # equivalent to i-1, j-1
    top_right = np.roll(top, -1, axis=1) # equivalent to i-1, j+1
    bottom_left = np.roll(bottom, 1, axis=1) # equivalent to i+1, j-1
    bottom_right = np.roll(bottom, -1, axis=1) # equivalent to i+1, j+1 

    # create boolean arrays for gradient orientations
    horizontal = (aprox_orientations == 0) | (aprox_orientations == 180) | (aprox_orientations == -180)
    vertical = (aprox_orientations == 90) | (aprox_orientations == -90)
    diagonal_right = (aprox_orientations == 45) | (aprox_orientations == -135)
    diagonal_left = ~(horizontal | vertical | diagonal_right) # all the rest that is not on the previous

    # apply maximun suppresion in all directions 
    edges = gradients.copy()  # Replace this line

    # only leave maximun in each direction
    edges[vertical & ((gradients < top) | (gradients < bottom))] = 0
    edges[horizontal & ((gradients < left) | (gradients < right))] = 0
    edges[diagonal_right & ((gradients < top_right) | (gradients < bottom_left))] = 0
    edges[diagonal_left & ((gradients < top_left) | (gradients < bottom_right))] = 0

    # zero-out borders (since rolled arrays wrap around)
    edges[[0, -1], :] = 0
    edges[:, [0, -1]] = 0

    ######################################################

    return edges
