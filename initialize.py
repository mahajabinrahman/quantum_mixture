#!/usr/bin/env python
import numpy as np
import random

def center_seed(k,data_set):
	centers = random.sample(data_set,k)
	return centers

def make_grid(array):
	void, array_0 = np.meshgrid([1],array)
	return array_0

def get_temp_cov(x,y,center):
	void,yy = np.meshgrid([1],y)
	void,xx = np.meshgrid([1],x)
	diff_x = xx-center[0]
	diff_y = yy-center[1]
	sigma_x = float(np.sqrt(sum(diff_x**2)/len(x)))
	sigma_y = float(np.sqrt(sum(diff_y**2)/len(y)))
	sigma_xy = float(sum(diff_x*diff_y)/len(x))
	diff_xy = xx - center[1]
	diff_yx = yy - center[0]
	sigma_yx = float(sum(diff_xy*diff_yx)/len(x))
	cov_matrix = np.matrix([[sigma_x**2,sigma_xy],[sigma_xy,sigma_y**2]])
	inv_cov = np.linalg.inv(cov_matrix)

	return inv_cov

def get_covariance(gamma,x_coord,y_coord,center,N_k):

	xx = make_grid(x_coord)
	yy = make_grid(y_coord)

	diff_x = xx - center[0]
	diff_y = yy - center[1]
	
	normalized_gamma = gamma/N_k

	sum(normalized_gamma)
	
       	sigma_x = float(np.sqrt(sum(normalized_gamma*diff_x**2)))
	sigma_y = float(np.sqrt(sum(normalized_gamma*diff_y**2)))
	sigma_xy = float(sum(normalized_gamma*diff_x*diff_y))
	

	cov_matrix = np.matrix([[sigma_x**2,sigma_xy],[sigma_xy,sigma_y**2]])

	return cov_matrix

def get_distribution(x_i,y_i,center,inv_cov):

	mu_x = center[0]
	mu_y = center[1]
	diff_matrix = np.matrix([[x_i - mu_x],[y_i - mu_y]])
	probability_distribution = np.exp(-(diff_matrix.transpose()*inv_cov*diff_matrix))
	return probability_distribution
	

def get_center(distribution,x,y):
	void,yy = np.meshgrid([1],y)
	void,xx = np.meshgrid([1],x)
      	weighted_x = distribution*xx
      	weighted_y = distribution*yy
	return weighted_x, weighted_y


def get_initial_guesses(k,data_set,centers,x_coord,y_coord):
	sums = []
	g_all_points = []
	probabilities = []
    
	for j in centers:
		inv_cov = get_temp_cov(x_coord, y_coord, j)
		point_distribution = []
		for coord in data_set: 
			g_i = get_distribution(coord[0],coord[1],j,inv_cov)
			g_i = float(g_i)
			point_distribution.append(g_i)
			
		g_all_points = make_grid(point_distribution)
		probabilities.append(g_all_points)

		sums.append(sum(g_all_points))
	


	weighted_gaussians=[]

	count = -1
	for j in centers:
		count += 1

		weight = sums[count]/sum(sums)
		distribution = probabilities[count]
		p_k = weight*distribution*(1.0/(sums[count]))
		weighted_gaussians.append(p_k)
		
	sum_p_k = sum(weighted_gaussians)


	N = []
	gammas = []

	for w in weighted_gaussians:
		p_k = w
		gamma = p_k*(1.0/sum_p_k)
		N_k = sum(gamma)
		gammas.append(gamma)	
		N.append(N_k)

	return gammas, N

def iterate_M(gammas,N,x_coord,y_coord):
	count=-1
	
	new_centers = []
	covariances = []
	weights = []

	total_points = sum(N)


	for gamma in gammas:
		count += 1
		N_k = N[count]
		weight = N_k/total_points
		weights.append(weight)
		normalized_gamma = gamma/N_k
		weighted_x, weighted_y = get_center(normalized_gamma,x_coord,y_coord)
		mu_x = (sum(weighted_x))
		mu_y = (sum(weighted_y))
		new_center = (round(float(mu_x),4),round(float(mu_y),4))
		new_centers.append(new_center)
		cov_matrix = get_covariance(gamma,x_coord,y_coord,new_center,N_k)
		covariances.append(cov_matrix)
		

	return new_centers,covariances,weights

def iterate_E(covariances,weights,mu,x_coord,y_coord,dataset):
	count = -1
	weighted_prob =[]
	covariance = []
	for covariance in covariances:
      		count += 1
		center = mu[count]
		weight = weights[count]
		cov_matrix = covariances[count]
		inv_cov = np.linalg.inv(cov_matrix)
		point_distribution=[]
		for coord in dataset:
			g_i = float(get_distribution(coord[0],coord[1],center,inv_cov)) 
			point_distribution.append(g_i)

		g_all_points = make_grid(point_distribution)
	
		probability = weight*g_all_points*(1.0/sum(g_all_points))
		
		weighted_prob.append(probability)
	
	sum_p_k = sum(weighted_prob)


	N = []
	gammas = []

	for p_k in weighted_prob:
		gamma = p_k*(1.0/sum_p_k)
		N_k = sum(gamma)
		gammas.append(gamma)	
		N.append(N_k)
	

	return gammas, N

