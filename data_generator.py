#!/usr/bin/env python

import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#randomly import parameters

def get_input(x_center,y_center, sigma_bound):
    theta = random.randint(0,360)
    theta = theta*(1.0/57.0)
    #sigma_x = random.randint(0,5)
    #sigma_y = random.randint(0,5)
    sigma_x = sigma_bound
    sigma_y = sigma_bound


    return theta,sigma_x,sigma_y,x_center,y_center

def get_prime_data(x_center,y_center, sigma_x, sigma_y): 

    x_prime = np.random.normal(x_center, sigma_x,100)
    y_prime = np.random.normal(y_center, sigma_y,100)

    return x_prime, y_prime 

def transform_coordinates(x_prime, y_prime,x_center,y_center, theta):

    coords = np.matrix([x_prime, y_prime])
    coords = coords.transpose()
    center_matrix = np.matrix([x_center, y_center])
    center_matrix = center_matrix.transpose()
    rotation_matrix = np.matrix([[np.cos(theta), np.sin(theta)],[(-1)*np.sin(theta), np.cos(theta)]])
    new_coords = rotation_matrix*coords
    abs_coords = new_coords - center_matrix

    return abs_coords 

def get_class_arrays(x_center,y_center,sigma_bound):

    theta, sigma_x, sigma_y, x_center, y_center  = get_input(x_center, y_center,sigma_bound)
    x_prime, y_prime = get_prime_data(x_center, y_center, sigma_x, sigma_y)
    x_coord = []
    y_coord = []
    for i in range(0,100):
        abs_coords = transform_coordinates(x_prime[i],y_prime[i],x_center,y_center, theta)
       

        x=x_prime[i]
        y=y_prime[i]

        x_coord.append(x)
        y_coord.append(y)
    
    return x_coord, y_coord, sigma_x, sigma_y









