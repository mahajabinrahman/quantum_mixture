#!/usr/bin/env python
import os, sys
import scipy
from scipy import optimize
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt


def get_temp_cov(x,y,mu_x, mu_y):
    void,yy = np.meshgrid([1],y)
    void,xx = np.meshgrid([1],x)
    diff_x = xx-mu_x
    diff_y = yy-mu_y
    var_x = float((sum(diff_x**2))/len(x))
    var_y = float((sum(diff_y**2))/len(y))
    var_xy = float((sum(diff_x*diff_y))/len(x))
    cov_matrix = np.matrix([[var_x,var_xy],[var_xy,var_y]])
    inv_cov = np.linalg.inv(cov_matrix)
    return cov_matrix

# initializes first iteration of parameters randomly
def get_initial_parameters(data_set,x_coord, y_coord,k):
    cos_phi = random.uniform(-1,1)
    centers = random.sample(data_set, k)
    center_1 = centers[0]
    center_2 = centers[1]
    alpha_1 = random.uniform(0,1)
    alpha_2 = random.uniform(0,1)
    cov_matrix_1 = get_temp_cov(x_coord,y_coord,center_1[0],center_1[1])
    cov_matrix_2 = get_temp_cov(x_coord,y_coord,center_2[0],center_2[1])
    return cos_phi, centers, cov_matrix_1, cov_matrix_2, alpha_1, alpha_2

#get gaussian component of the di
def get_gaussian(cov_matrix, data_set, mu_x, mu_y):
    inv_cov = np.linalg.inv(cov_matrix)
    gaussian_distribution_array = np.zeros((200,))
    count = 0
    for coords in data_set:
        x_i = coords[0]
        y_i = coords[1]

        diff_matrix = np.matrix([[x_i - mu_x],[y_i - mu_y]])
        expectation = np.exp(-(1.0/4)*diff_matrix.transpose()*inv_cov*diff_matrix)
        gaussian_distribution_array[count] = expectation
        count += 1
 
    normalization = np.sqrt(sum(gaussian_distribution_array**2))
    gaussian_distribution_array = (1/normalization)*gaussian_distribution_array   
    return gaussian_distribution_array

def get_distribution(data_set, cos_phi, cov_matrix_1, cov_matrix_2,mu_x0,mu_y0,mu_x1,mu_y1,alpha_1,alpha_2):
    # computing membership weights for each point belonging in a class.

    G_1 = get_gaussian(cov_matrix_1, data_set,mu_x0,mu_y0)
    G_2 = get_gaussian(cov_matrix_2, data_set,mu_x1,mu_y1)

    E_1 = (alpha_1**2)*G_1**2 + alpha_1*alpha_2*cos_phi*G_1*G_2
    E_2 = (alpha_2**2)*G_2**2 + alpha_1*alpha_2*cos_phi*G_1*G_2
    
    E_1 = E_1/sum(E_1)
    E_2 = E_2/sum(E_2) 
    
    Q_1 = E_1/(E_1 + E_2)
    Q_2 = E_2/(E_1 + E_2)

    return Q_1, Q_2

def get_cos_phi(alpha_1, alpha_2, G_1, G_2):
    cos_phi = (1 - alpha_1**2 - alpha_2**2)/(2*alpha_2*alpha_1*sum(G_1*G_2))
    return cos_phi

def get_lambda(cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2):
    alpha = alpha_1/alpha_2
    gamma = G_1/G_2

    factor = 1/(2*alpha_1*alpha_2*sum(G_1*G_2))
    lambda_0 = sum(Q_1/(alpha*gamma + cos_phi) + Q_2/((alpha**(-1))*(gamma**(-1))+cos_phi))
    lambda_0 = factor*lambda_0
	
    return lambda_0


def get_alpha_1(lambda_0, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2):

    fun = lambda alpha_1: sum(-Q_1*np.log(alpha_1**2 + alpha_1*alpha_2*G_1*G_2*cos_phi) - Q_2*np.log(alpha_2**2 + alpha_1*alpha_2*G_1*G_2*cos_phi))

    jac = lambda alpha_1: sum(-Q_1*(2*alpha_1*(G_1/G_2) + alpha_2*cos_phi)/(alpha_1*(alpha_1*(G_1/G_2) + alpha_2*cos_phi)) - Q_2*(cos_phi/(alpha_2*(G_2/G_1) + alpha_1*cos_phi)))

    x0 = [1.0]

    value = optimize.minimize(fun, x0, method='CG', jac=jac)
#    alpha_1 = float(value.x)

#    gamma = G_1/G_2
#
#    alpha_1 = sum(Q_1*(cos_phi)/(alpha_1*gamma + alpha_2*cos_phi) + Q_2*(2*alpha_2*(gamma**(-1)) + alpha_1*cos_phi)/(alpha_2*(alpha_2*gamma**(-1) + alpha_1*cos_phi))) - 2*lambda_0*alpha_2**2
    
#    alpha_1 = alpha_1/(2*lambda_0*sum(G_1*G_2)*cos_phi)

    alpha_1 = -.9
    return alpha_1

def get_alpha_2(lambda_0, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2):

    fun = lambda alpha_2: -sum(Q_1*np.log(alpha_1**2 + alpha_1*alpha_2*G_1*G_2*cos_phi) + Q_2*np.log(alpha_2**2 + alpha_1*alpha_2*G_1*G_2*cos_phi))

    jac = lambda alpha_2: -sum(Q_1*(cos_phi)/(alpha_1*(G_1/G_2)) + Q_2*((2*alpha_2*(G_2/G_1) + alpha_1*cos_phi)/(alpha_2*(alpha_2*(G_2/G_1) + alpha_1*cos_phi))) - 2*lambda_0*(alpha_2 + alpha_1*G_1*G_2*cos_phi))

    x0 = [1.2]

    value = optimize.minimize(fun,x0, method='CG',jac=jac)
    
#    alpha_2 = float(value.x)
#    gamma = G_1/G_2

#    alpha_2 = sum(Q_1*(cos_phi)/(alpha_1*gamma + alpha_2*cos_phi) + Q_2*(2*alpha_2*(gamma**(-1)) + alpha_1*cos_phi)/(alpha_2*(alpha_2*gamma**(-1) + alpha_1*cos_phi)) - 2*lambda_0*alpha_1*G_1*G_2*cos_phi)
    #
#    alpha_2 = alpha_2/(2*lambda_0)

    alpha_2 = .8

    return alpha_2

def get_mu_1(x, y, lambda_0, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2):

    alpha_gamma = (alpha_1/alpha_2)*(G_1/G_2)

    F_i  = Q_1*((alpha_gamma + (1.0/2)*cos_phi)/(alpha_gamma+cos_phi)) + Q_2*((1.0/2)*cos_phi/((alpha_gamma)**(-1)+cos_phi)) - lambda_0*alpha_1*alpha_2*cos_phi*G_1*G_2

    mu_x = round(float(sum(F_i*x)/sum(F_i)),4)
    mu_y = round(float(sum(F_i*y)/sum(F_i)),4)
    new_center_1 = (mu_x, mu_y)
    
    return new_center_1


def get_mu_2(x,y,lambda_0, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2):
 
    alpha_gamma = (alpha_1/alpha_2)*(G_1/G_2)
    
    F_i  = Q_2*((alpha_gamma**(-1) + (1.0/2)*cos_phi)/(alpha_gamma**(-1)+ (1.0/2)*cos_phi)) + Q_1*((1.0/2)*cos_phi/((alpha_gamma)+cos_phi)) - lambda_0*alpha_1*alpha_2*cos_phi*G_1*G_2
    
    mu_x = round(float(sum(F_i*x)/sum(F_i)),4)
    mu_y = round(float(sum(F_i*y)/sum(F_i)),4)
    new_center_2 = (mu_x, mu_y)
    
    return new_center_2

def cov_matrix_1(lambda_0, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2, x, y,  mu_1):
    mu_x = mu_1[0]
    mu_y = mu_1[1]
    diff_x = x - mu_x
    diff_y = y - mu_y

    H_i  = Q_1*(2*alpha_1*G_1 + alpha_2*G_2*cos_phi)/(alpha_1*G_1 + alpha_2*G_2*cos_phi) + Q_2*(alpha_1*G_1*cos_phi)/(alpha_2*G_2 + alpha_1*G_1*cos_phi)

    H_i = H_i - lambda_0*(alpha_1*G_1*(2*alpha_1*G_1 + 2*alpha_2*G_2*cos_phi))

    normalized_H_i = H_i/sum(H_i)

    var_x = float(sum(normalized_H_i*diff_x**2))
    var_y = float(sum(normalized_H_i*diff_y**2))
    var_xy = float(sum(normalized_H_i*diff_x*diff_y))
	
    cov_matrix_1 = np.matrix([[var_x,var_xy],[var_xy,var_y]])
    
    if var_x*var_y - var_xy**2 < 1.0e-10:
        var_xy**2 == 1.0e-11

#    print'COV 1', (normalized_H_i.sum(), (diff_x**2).sum()), alpha_1, alpha_2

#    for (a, b) in zip(normalized_H_i, diff_x**2):
#        print(a, b), a*b
#    print(var_x, var_y, var_xy, np.linalg.det(cov_matrix_1), var_x*var_y - var_xy**2)

    return cov_matrix_1

def cov_matrix_2(lambda_0, cos_phi, Q_1, Q_2, G_1, G_2, alpha_1, alpha_2, x, y, mu_2):

    mu_x = mu_2[0]
    mu_y = mu_2[1]

    diff_x = x - mu_x
    diff_y = y - mu_y

    H_i  = Q_1*(alpha_2*G_2*cos_phi)/(alpha_1*G_1 + alpha_2*G_2*cos_phi) + Q_2*(2*alpha_2*G_2 + alpha_1*G_1*cos_phi)/(alpha_2*G_2 + alpha_1*G_1*cos_phi)

    H_i = H_i - lambda_0*(alpha_2*G_2*(2*alpha_2*G_2 + 2*alpha_1*G_1*cos_phi))

    normalized_H_i = H_i/sum(H_i)


    var_x = float(sum(normalized_H_i*diff_x))
    var_y = float(sum(normalized_H_i*diff_y))
    var_xy = float(sum(H_i*diff_x*diff_y))
	
    cov_matrix_2 = np.matrix([[var_x,var_xy],[var_xy,var_y]])
    if var_x*var_y - var_xy**2 < 1.0e-10:
        var_xy**2 == 1.0e-11
    

#    print'COV 2', (normalized_H_i.sum(), (diff_x**2).sum()), cos_phi
#    for (a, b) in zip(normalized_H_i, diff_x**2):
#        print(a, b), a*b
#    print(var_x, var_y, var_xy, np.linalg.det(cov_matrix_1), var_x*var_y - var_xy**2)


    return cov_matrix_2


#corrections
#check gaussian normalization
#order?
#check alpha
#check covariance
#fix expectation value calculation
    