import os, sys
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import data_generator as ds
import calculate_param as cm
import csv 

random.seed(412)
np.random.seed(78)

x_coord,y_coord,real_cov1, real_cov2, real_x1, real_y1, real_x2, real_y2 = ds.get_data(5,10, 20,12,4,10)

data_set = zip(x_coord,y_coord)
x_coord = np.array(x_coord)
y_coord = np.array(y_coord)

plt.scatter(x_coord, y_coord)

cos_phi, centers, cov_matrix_1, cov_matrix_2, alpha_1, alpha_2 = cm.get_initial_parameters(data_set, x_coord, y_coord,2)
mu_x = centers[0][0]
mu_y = centers[0][1]
mu_x1 = centers[1][0]
mu_y1 = centers[1][1]

new_centers = []

G_1 = cm.get_gaussian(cov_matrix_1, data_set, mu_x, mu_y)
G_2 = cm.get_gaussian(cov_matrix_2, data_set, mu_x1,mu_y1)


cos_phi = cm.get_cos_phi(alpha_1, alpha_2, G_1,G_2)
Q_1,Q_2 = cm.get_distribution(data_set,cos_phi,cov_matrix_1, cov_matrix_2, mu_x, mu_y, mu_x1, mu_y1,alpha_1,alpha_2)
lambda_nc = float(cm.get_lambda(cos_phi, Q_1, Q_2, G_1,G_2, alpha_1,alpha_2))
alpha_1 = cm.get_alpha_1(lambda_nc, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2)
alpha_2 = cm.get_alpha_2(lambda_nc, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2)
new_center_1 = cm.get_mu_1(x_coord, y_coord, lambda_nc, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2) 
new_center_2 = cm.get_mu_2(x_coord, y_coord, lambda_nc, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2)

cov_matrix_1 = cm.cov_matrix_1(lambda_nc, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2, x_coord, y_coord,new_center_1)
cov_matrix_2 = cm.cov_matrix_2(lambda_nc, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2, x_coord,y_coord, new_center_2)

#G = G_1*G_2*cos_phi

#A =  sum(-Q_1*((2*alpha_1 + G*alpha_2)**2)/(alpha_1**2 + G*alpha_2*alpha_1)**2 + 2*Q_1/(alpha_1**2 + G*alpha_1*alpha_2) - (G**2)*Q_2*(alpha_2**2)/((G*alpha_1*alpha_2 + alpha_2**2)**2) - 2*lambda_nc)

#B = sum(-(G**2)*Q_1*(alpha_2**2)/((alpha_1**2 + G*alpha_1*alpha_2)**2) - Q_1*((G*alpha_1 + 2*alpha_2)**2)/((G*alpha_1*alpha_2 + alpha_2**2)**2) + 2*Q_2/(G*alpha_1*alpha_2 + alpha_2**2) - 2*lambda_nc)

#C = sum(-(G*Q_1*alpha_1*(2*alpha_1 + G*alpha_2))/((alpha_1**2 + G*alpha_1*alpha_2)**2) + (G*Q_1)/(alpha_1**2 + G*alpha_1*alpha_2) + (G*Q_2)/(alpha_2**2 + G*alpha_1*alpha_2) - (G*Q_2*alpha_2*(2*alpha_2 + G*alpha_1))/((alpha_2**2 + G*alpha_1*alpha_2)**2) - 2*G*lambda_nc)

#det = A*B - C**2

#print "determinant",det


new_centers = [new_center_1, new_center_2]

while set(centers) != set(new_centers):
    centers = new_centers
    mu_x = centers[0][0]
    mu_y = centers[0][1]
    mu_x1 = centers[1][0]
    mu_y1 = centers[1][1]
    G_1 = cm.get_gaussian(cov_matrix_1, data_set, mu_x, mu_y)
    G_2 = cm.get_gaussian(cov_matrix_2, data_set, mu_x1,mu_y1)
    cos_phi = cm.get_cos_phi(alpha_1, alpha_2, G_1,G_2)
    print 'sum of G', sum(G_1*G_2), sum(G_1**2), alpha_1, alpha_2, cos_phi, lambda_nc, centers, cov_matrix_1, cov_matrix_2
    Q_1,Q_2 = cm.get_distribution(data_set,cos_phi,cov_matrix_1, cov_matrix_2, mu_x, mu_y, mu_x1, mu_y1,alpha_1,alpha_2)
    
    lambda_nc = float(cm.get_lambda(cos_phi, Q_1, Q_2, G_1,G_2, alpha_1,alpha_2))
    alpha_1 = cm.get_alpha_1(lambda_nc, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2)
    alpha_2 = cm.get_alpha_2(lambda_nc, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2)
    new_center_1 = cm.get_mu_1(x_coord, y_coord, lambda_nc, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2)
    new_center_2 = cm.get_mu_2(x_coord, y_coord, lambda_nc, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2)
    cov_matrix_1 = cm.cov_matrix_1(lambda_nc, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2, x_coord, y_coord,new_center_1)
    
    cov_matrix_2 = cm.cov_matrix_2(lambda_nc, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2, x_coord,y_coord, new_center_2)
    
    new_centers = [new_center_1, new_center_2]
    



#plt.scatter([x[0] for x in new_centers],[x[1] for x in new_centers],s=50, c=u'c', marker=u'o',cmap=None, norm=None, vmin=None, vmax=50, alpha=None, linewidths=None, verts=None, hold=None)

#plt.show()
