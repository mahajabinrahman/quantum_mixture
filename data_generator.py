#!/usr/bin/env python

import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

#randomly import parameters




#generates data points by sampling  gaussian about the origin
def get_sampled_data(sigma_x, sigma_y, n=100): 

    x_prime = np.random.normal(0, sigma_x,n)
    y_prime = np.random.normal(0, sigma_y,n)

    return x_prime, y_prime 

def transform_coordinates(x_prime, y_prime,x_center,y_center, theta):

    coords = np.matrix([x_prime, y_prime])
    center_matrix = np.matrix([[x_center], [y_center]])
    rotation_matrix = np.matrix([[np.cos(theta), np.sin(theta)],[(-1)*np.sin(theta), np.cos(theta)]])
    new_coords = rotation_matrix*coords
    abs_coords = new_coords - center_matrix

    return abs_coords 

def get_class_arrays(x_center,y_center,sigma_x, sigma_y,n=100):

    theta = random.random()*2*math.pi
    x_prime, y_prime = get_sampled_data(sigma_x, sigma_y,n)
    x_coord = []
    y_coord = []
    
    abs_coords = transform_coordinates(x_prime, y_prime, x_center, y_center, theta)
    x_coord = abs_coords[0]
    y_coord = abs_coords[1]
    
    
    return x_coord, y_coord


def get_data(mu_x1, mu_y1, mu_x2, mu_y2, sigma_x, sigma_y):
    x_coord, y_coord = get_class_arrays(mu_x1, mu_y1, sigma_x, sigma_y)
    x_coord_1 = np.array(x_coord)
    y_coord_1 = np.array(y_coord)
   
    coords_1 = np.array([x_coord_1[0], y_coord_1[0]])
    cov_matrix_1 = np.cov(coords_1)

    x_coord2, y_coord2 = get_class_arrays(mu_x2, mu_y2, sigma_x, sigma_y)
    x_coord_2 = np.array(x_coord2)
    y_coord_2 = np.array(y_coord2)
   
    coords_2 = np.array([x_coord_2[0], y_coord_2[0]])
    cov_matrix_2 = np.cov(coords_2)
    
    x_coord = x_coord_1[0].tolist()
    y_coord = y_coord_1[0].tolist()
    x_coord2 = x_coord_2[0].tolist()
    y_coord2 = y_coord_2[0].tolist()
   
    x_coord.extend(x_coord2)
    y_coord.extend(y_coord2)
    return x_coord, y_coord, cov_matrix_1, cov_matrix_2, mu_x1, mu_y1, mu_x2, mu_y2







