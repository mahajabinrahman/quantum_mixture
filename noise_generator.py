#!/usr/bin/env python

import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



def new_points(range, x_array, y_array):
    new_x_array = []
    new_y_array = []
    for j in x_array: 
        error_x = random.randint(-2,-2)
        new_x = j + error_x
        new_x_array.append(new_x)
        
        error_y = random.randint(-2,2)
        new_y = y_array[x_array.index(j)] + error_y 
        new_y_array.append(new_y)
        
        return new_x_array, new_y_array



    
        
