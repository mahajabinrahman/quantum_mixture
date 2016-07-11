#!/usr/bin/env python
import os, sys
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import dataset_creator as ds
import initialize as initialize

results = open("output.txt", 'w')

for epsilon in range(0,5):
	epsilon = str(epsilon)
	results.write(epsilon)


x_coord,y_coord,real_cov1, real_cov2, real_x1, real_y1, real_x2, real_y2 = ds.get_data(5,10, 20,12,2)

#for epsilon in range(0,5):
#	new_x, new_y = ds.get_epsilon(epsilon, x_coord, y_coord)
#	x_coord = new_x
#	y_coord = new_y
data_set = zip(x_coord,y_coord)

plt.scatter(x_coord,y_coord)
k = 2

centers = initialize.center_seed(k,data_set)
new_centers = []

gammas, N = initialize.get_initial_guesses(k,data_set,centers,x_coord,y_coord)

new_centers,covariances,weights = initialize.iterate_M(gammas,N, x_coord, y_coord)


while set(centers) != set(new_centers):
	centers = new_centers 
	new_centers = []
	gammas, N = initialize.iterate_E(covariances, weights, centers, x_coord,y_coord,data_set)
	new_centers,covariances,weights = initialize.iterate_M(gammas,N,x_coord, y_coord)
	print covariances

	
		
plt.scatter([x[0] for x in new_centers],[x[1] for x in new_centers],s=50, c=u'c', marker=u'o',cmap=None, norm=None, vmin=None, vmax=50, alpha=None, linewidths=None, verts=None, hold=None)


plt.show()




results.close()





