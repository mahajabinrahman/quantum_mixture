#!/usr/bin/env python
import os
import sys
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import data_generator as ds
import initialize as initialize


random.seed(400)
np.random.seed(78)



x_coord, y_coord, real_cov1, real_cov2, mu_x1, mu_y1, mu_x2, mu_y2 = ds.get_data(mu_x1=-10, mu_y1=-10, mu_x2=10, mu_y2=10, sigma_x=5, sigma_y=2)
print real_cov1
print real_cov2
print  mu_x1, mu_y1, mu_x2, mu_y2

data_set = zip(x_coord, y_coord)
x_coord = np.array(x_coord)
y_coord = np.array(y_coord)
plt.scatter(x_coord, y_coord)
plt.xlim([-20, 20])
plt.ylim([-20, 20])
k = 2

centers = initialize.center_seed(k, data_set)
new_centers = []

gammas, N = initialize.get_initial_guesses(
    k, data_set, centers, x_coord, y_coord)


new_centers, covariances, weights = initialize.iterate_M(
    gammas, N, x_coord, y_coord)


while not np.allclose(centers, new_centers, rtol=1e-05, atol=1e-08):
    centers = new_centers
    new_centers = []
    gammas, N = initialize.iterate_E(covariances, weights, centers,
                                     x_coord, y_coord, data_set)
    new_centers, covariances, weights = initialize.iterate_M(
        gammas, N, x_coord, y_coord)

print "centers:", new_centers
print "covariance 1:", covariances[0]
print "covariance 2:",  covariances[1]
print " weights:", weights


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
