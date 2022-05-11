import random

import numpy
import numpy as np
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt

data = datasets.load_iris().data


def kmeans(data, ncluster):
    centroids = init_centroids(ncluster, data)
    error = np.array([])
    improving = True
    i = 0
    while improving:
        centroid_assign, iter_error = assign_centroid(data, centroids)
        # print(centroid_assign)
        error = np.append(error, np.sum(iter_error))
        # print("error", error, i)
        # print("assign", centroid_assign)
        break
        new_centroids = []
        for j in range(ncluster):
            new_centroid1 = calculate_mean_of_data(centroid_assign, data, j)
            if new_centroid1 != None:
                new_centroids.append(new_centroid1)
            else:
                new_centroids.append(centroids[j])
        centroids = np.array(new_centroids)
        # plt.scatter(data[:, 0], data[:, 1], marker='o')
        # plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=300)
        # plt.show()
        # print("new centroids iter", centroids, error)
        if (len(error)<2):
            pass
        else:
            if i < len(error):
                if (round(error[i-1],3) == round(error[i], 3)):
                    improving = False
        i += 1
    centroid_assign, iter_error = assign_centroid(data, centroids)
    # print(centroid_assign, iter_error)
    # print("centroids", centroids)
    new_centroids = []
    for p in range(ncluster):
        new_centroid1 = calculate_mean_of_data(centroid_assign, data, p)
        if new_centroid1 != None:
            new_centroids.append(new_centroid1)
        else:
            new_centroids.append(centroids[p])
    centroids = np.array(new_centroids)
    # print("new centroid list ende", centroids, "itererror", iter_error)
    return (centroid_assign, iter_error, centroids, error)
def calculate_mean_of_data(centroid_assign, data, centroid_number):
    filterassign = []
    for index, a in enumerate(centroid_assign):
        if a == centroid_number:
            filterassign.append(True)
        else:
            filterassign.append(False)
    new_data = data[filterassign]
    new_centroid = []
    for i in range(new_data.shape[1]):  # normalizing every column separately
        if len(new_data[:, i]) != 0:
            new_centroid.append(np.mean(new_data[:, i]))  # change current variable to normalized data
        else:
            new_centroid.append(0)
    # print("n_centroid", new_centroid)
    # print("mean data", new_data)
    return new_centroid

def euclideandistance(data1, data2):
    distance = np.sqrt(np.sum((data1 - data2) ** 2))
    print("data", data1, data2)
    print("distance", distance)
    return distance

def init_centroids(k, data):
    n_indices = np.random.choice(range(data.shape[0]), size=k, replace=False)
    centroids = [data[x] for x in n_indices]
    # numberdim = data.shape[1]
    # centroid_min = np.min(data)
    # centroid_max = np.max(data)
    # centroids = []
    # for i in range(k):
    #     centroid = np.random.uniform(centroid_min, centroid_max, numberdim)
    #     # centroid = np.random.uniform(0, 1000, numberdim)
    #     centroids.append(centroid)
    centroids = np.array(centroids)
    # plt.scatter(data[:, 0], data[:, 1], marker='o')
    # plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=300)
    # plt.show()
    return centroids

def assign_centroid(data, centroids):
    dimension = data.shape[0]
    centroid_assign = []
    # centroid_errors = []
    k = centroids.shape[0]
    # for i in range(dimension):
    #     errors = np.array([])
    #     for centroid_index in range(k):
    #         distance = euclideandistance(centroids[centroid_index], data[i])
    #         errors = np.append(errors, distance)
    #     closest_centroid = np.where(errors == np.amin(errors))[0].tolist()[0]
    #     centroid_error = np.amin(errors)
    #     centroid_assign.append(closest_centroid)
    #     centroid_errors.append(centroid_error)
    #     # print("errors", centroid_errors)
    errors = np.array([])
    for centroid_ind in range(k):
        distance = euclideandistance(centroids[centroid_ind], data)
        errors = np.append(errors, [distance])
    print("error", errors)
    return (centroid_assign, errors)
error_lst = []
for i in range(1,2):
    centroids, error, iter_error = kmeans(data, i)[2], kmeans(data, i)[3], kmeans(data, i)[1]
    print(len(iter_error), i)
    # centroids = kmeans(data, i)[2]
    plt.scatter(data[:,0], data[:,1],  marker = 'o')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', color='red', s=300)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.show()
    # plt.scatter(data[:, 0], data[:, 3], marker='o')
    # plt.scatter(centroids[:, 0], centroids[:, 3], marker='o', color='red', s=300)
    # plt.xlabel('Sepal length')
    # plt.ylabel('Petal width')
    # print(iter_error, "iter error")
    # print(error, "error")
    error_lst.append(np.sum(iter_error))
    # plt.plot([1, 2], [np.sum(iter_error), np.sum(iter_error)])
    # plt.xlabel(str(i)+' Cluster')
    # plt.ylabel('Error')
    # plt.show()
print(min(error_lst), error_lst.index(min(error_lst)))

