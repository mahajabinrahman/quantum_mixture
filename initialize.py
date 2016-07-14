#!/usr/bin/env python
import numpy as np
import random


def center_seed(k, data_set):
    centers = random.sample(data_set, k)
    return centers


def get_covariance(gamma, x, y, center, N_k):
    diff_x = x - center[0]
    diff_y = y - center[1]

    normalized_gamma = gamma / N_k

    var_x = float(sum(normalized_gamma * diff_x**2))
    var_y = float(sum(normalized_gamma * diff_y**2))
    var_xy = float(sum(normalized_gamma * diff_x * diff_y))

    cov_matrix = np.matrix([[var_x, var_xy], [var_xy, var_y]])

    return cov_matrix


def get_distribution(x_i, y_i, center, inv_cov):
    mu_x = center[0]
    mu_y = center[1]
    diff_matrix = np.matrix([[x_i - mu_x], [y_i - mu_y]])
    probability_distribution = np.exp(-(1 / 2.0) *
                                      (diff_matrix.transpose() * inv_cov * diff_matrix))
    return probability_distribution


def get_center(distribution, x, y):
    weighted_x = distribution * x
    weighted_y = distribution * y
    return weighted_x, weighted_y


def get_initial_guesses(k, data_set, centers, x_coord, y_coord):
    sums_list = []
    point_distribution_array = np.zeros((2, 200))

    count_j = 0

    for j in centers:
        cov = np.cov([x_coord, y_coord])
        inv_cov = np.linalg.inv(cov)

        count_i = 0

        for coord in data_set:

            g_i = float(get_distribution(coord[0], coord[1], j, inv_cov))
            point_distribution_array[count_j, count_i] = g_i
            count_i += 1

        sums_list.append(sum(point_distribution_array[count_j]))
        count_j += 1

    weighted_gaussians = []

    count = 0
    for j in centers:
        weight = sums_list[count] / sum(sums_list)
        distribution = point_distribution_array[count]
        p_k = weight * distribution * (1.0 / (sums_list[count]))
        weighted_gaussians.append(p_k)
        count += 1
    sum_p_k = sum(weighted_gaussians)

    N_points = []
    gammas = []

    for w in weighted_gaussians:
        p_k = w
        gamma = p_k * (1.0 / sum_p_k)
        N_k = sum(gamma)
        gammas.append(gamma)
        N_points.append(N_k)

    return gammas, N_points


def iterate_M(gammas, N, x_coord, y_coord):
    count = 0

    new_centers = []
    covariances = []
    weights = []

    total_points = sum(N)

    for gamma in gammas:

        N_k = N[count]
        weight = N_k / total_points
        weights.append(weight)
        normalized_gamma = gamma / N_k
        weighted_x, weighted_y = get_center(normalized_gamma, x_coord, y_coord)
        mu_x = (sum(weighted_x))
        mu_y = (sum(weighted_y))
        new_center = (float(mu_x), float(mu_y))
        new_centers.append(new_center)
        cov_matrix = get_covariance(gamma, x_coord, y_coord, new_center, N_k)
        covariances.append(cov_matrix)
        count += 1

    return new_centers, covariances, weights


def iterate_E(covariances, weights, mu, x_coord, y_coord, dataset):

    weighted_prob = []
    covariance = []
    point_distribution_array = np.zeros((2, 200))

    count_j = 0
    for covariance in covariances:
        center = mu[count_j]
        weight = weights[count_j]
        cov_matrix = covariances[count_j]
        inv_cov = np.linalg.inv(cov_matrix)

        count_i = 0
        for coord in dataset:
            g_i = float(get_distribution(coord[0], coord[1], center, inv_cov))
            point_distribution_array[count_j, count_i] = g_i
            count_i += 1
        probability = weight * \
            point_distribution_array[count_j] * (1.0 / sum(point_distribution_array[count_j]))
        count_j += 1

        weighted_prob.append(probability)

    sum_p_k = sum(weighted_prob)

    N_points = []
    gammas = []

    for p_k in weighted_prob:
        gamma = p_k * (1.0 / sum_p_k)
        N_k = sum(gamma)
        gammas.append(gamma)
        N_points.append(N_k)

    return gammas, N_points
