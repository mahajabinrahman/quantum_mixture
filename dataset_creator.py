#!/usr/bin/env python

import noise_generator as error
import data_generator as data
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

def get_covariance(x,y,mu_x, mu_y):
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



def get_data(mu_x1,mu_y1,mu_x2,mu_y2,error_range):
    x_coord, y_coord,sigx1,sigy1 = data.get_class_arrays(mu_x1,mu_y1,error_range)
    cov_matrix_1 = get_covariance(x_coord, y_coord, mu_x1,mu_y1) 

    x_coord2,y_coord2,sigx2,sigy2 = data.get_class_arrays(mu_x2,mu_y2,error_range)
    cov_matrix_2 = get_covariance(x_coord2,y_coord2, mu_x2, mu_y2)


    x_coord.extend(x_coord2)
    y_coord.extend(y_coord2)
        

#new_x_coords, new_y_coords = error.new_points(error_range, x_coord, y_coord)
    return x_coord, y_coord,cov_matrix_1, cov_matrix_2, mu_x1, mu_y1, mu_x2, mu_y2

def get_epsilon(epsilon, x_array, y_array):
    new_x_array = []
    new_y_array = []
    for j in x_array: 
        error_x = random.randint(-2,-2)
        new_x = j + error_x
        new_x_array.append(new_x)
    for k in y_array:
        error_y = random.randint(-2,2)
        new_y = y_array[x_array.index(j)] + error_y 
        new_y_array.append(new_y)
        
        return new_x_array, new_y_array
