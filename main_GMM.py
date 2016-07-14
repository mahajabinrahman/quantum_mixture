#!/usr/bin/env python
import os
import sys
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import data_generator as ds
import initialize as initialize


x_coord, y_coord, real_cov1, real_cov2, real_x1, real_y1, real_x2, real_y2 = ds.get_data(
    5, 10, 11, 16, 4, 2)

data_set = zip(x_coord, y_coord)
x_coord = np.array(x_coord)
y_coord = np.array(y_coord)
plt.scatter(x_coord, y_coord)
k = 2

centers = initialize.center_seed(k, data_set)
new_centers = []

gammas, N = initialize.get_initial_guesses(
    k, data_set, centers, x_coord, y_coord)


new_centers, covariances, weights = initialize.iterate_M(
    gammas, N, x_coord, y_coord)


if np.allclose(centers, new_centers, rtol=1e-05, atol=1e-08) == False:
    centers = new_centers
    new_centers = []
    gammas, N = initialize.iterate_E(covariances, weights, centers,
                                     x_coord, y_coord, data_set)
    new_centers, covariances, weights = initialize.iterate_M(
        gammas, N, x_coord, y_coord)

    print new_centers


plt.scatter([x[0] for x in new_centers],
            [x[1] for x in new_centers],
            s=50,
            c=u'c',
            marker=u'o',
            cmap=None,
            norm=None,
            vmin=None,
            vmax=50,
            alpha=None,
            linewidths=None,
            verts=None,
            hold=None)


plt.show()
