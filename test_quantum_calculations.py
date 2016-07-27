import os, sys
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import data_generator as ds
import calculate_param as cm

random.seed(400)
np.random.seed(78)

x_coord,y_coord,real_cov1, real_cov2, real_x1, real_y1, real_x2, real_y2 = ds.get_data(5,10, 10,12,7,4)

data_set = zip(x_coord,y_coord)
x_coord = np.array(x_coord)
y_coord = np.array(y_coord)

# get initial parameters
cos_phi, centers, cov_matrix_1, cov_matrix_2, alpha_1, alpha_2 = cm.get_initial_parameters(data_set, x_coord, y_coord,2)

def E_Step(centers, cov_matrix_1, cov_matrix_2, data_set, alpha_1, alpha_2): 
    mu_x = centers[0][0]
    mu_y = centers[0][1]
    mu_x1 = centers[1][0]
    mu_y1 = centers[1][1]

    G_1 = cm.get_gaussian(cov_matrix_1, data_set, mu_x, mu_y)
    G_2 = cm.get_gaussian(cov_matrix_2, data_set, mu_x1,mu_y1)
    Q_1,Q_2 = cm.get_distribution(data_set,cos_phi,cov_matrix_1, cov_matrix_2, mu_x, mu_y, mu_x1, mu_y1,alpha_1,alpha_2)

    return G_1,G_2, Q_1, Q_2

def M_Step(Q_1,Q_2, G_1, G_2, alpha_1, alpha_2):
    cos_phi = cm.get_cos_phi(alpha_1, alpha_2, G_1,G_2)
    lambda_nc = float(cm.get_lambda(cos_phi, Q_1, Q_2, G_1,G_2, alpha_1,alpha_2))
    alpha_1, alpha_2= cm.get_alphas(lambda_nc, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2)
    new_center_1 = cm.get_mu_1(x_coord, y_coord, lambda_nc, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2) 
    new_center_2 = cm.get_mu_2(x_coord, y_coord, lambda_nc, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2)
    new_centers = [new_center_1, new_center_2]
    cov_matrix_1 = cm.cov_matrix_1(lambda_nc, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2, x_coord, y_coord,new_center_1)
    cov_matrix_2 = cm.cov_matrix_2(lambda_nc, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2, x_coord,y_coord, new_center_2)

    return cos_phi, lambda_nc, alpha_1, alpha_2, new_centers, cov_matrix_1, cov_matrix_2

# get initial parameters
cos_phi, centers, cov_matrix_1, cov_matrix_2, alpha_1, alpha_2 = cm.get_initial_parameters(data_set, x_coord, y_coord,2)

G_1, G_2, Q_1, Q_2 = E_Step(centers, cov_matrix_1, cov_matrix_2, data_set, alpha_1, alpha_2)
cos_phi, lambda_nc, alpha_1, alpha_2, new_centers, cov_matrix_1, cov_matrix_2 = M_Step(Q_1,Q_2, G_1, G_2, alpha_1, alpha_2)


while set(centers) != set(new_centers):
    centers = new_centers
    G_1, G_2, Q_1, Q_2 = E_Step(centers, cov_matrix_1, cov_matrix_2, data_set, alpha_1, alpha_2)
    cos_phi, lambda_nc, alpha_1, alpha_2, new_centers, cov_matrix_1, cov_matrix_2 = M_Step(Q_1,Q_2, G_1, G_2, alpha_1, alpha_2)
    print 'cos phi:', cos_phi
    print 'alpha_1 & alpha_2', alpha_1, alpha_2
    print 'probability sum', alpha_1**2 + alpha_2**2 + 2*alpha_1*alpha_2*cos_phi*sum(G_1*G_2)
    print 'cov 1', cov_matrix_1
    print 'cov 2', cov_matrix_2
    




