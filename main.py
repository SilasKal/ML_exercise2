import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

iris_data = datasets.load_iris().data


def kmeans(data, ncluster):
    centroids = init_centroids(ncluster, data)  # initialize centroids by randomly creating them
    assignments = []  # list for checking whether assignment changes
    improving = True
    while improving:
        centroid_assign, dist_matrix = assign_centroid(data,
                                                       centroids)  # assign data points to centroids and calculate distance matrix
        assignments.append(centroid_assign)  # append current assignment to assignmentlist
        new_centroids = np.array([])  # intialize array to save new centroids
        for i in range(ncluster):  # for every centroid relocate by calculating mean of assigned data
            new_centroid = calculate_mean_of_data(centroid_assign, data, i)
            new_centroids = np.append(new_centroids, new_centroid)
        if len(assignments) > 1:  # checking whether assignment still changes
            if assignments[0].all() == assignments[1].all():
                improving = False
            else:
                assignments = assignments[1:]
    return centroids, dist_matrix, centroid_assign


def calculate_mean_of_data(centroid_assign, data, centroid_number):
    filterassign = []
    for a in centroid_assign:  # creating filter list to filter data by assignment to centroid
        if a == centroid_number:
            filterassign.append(True)
        else:
            filterassign.append(False)
    new_data = data[filterassign]  # data assigned to current centroid
    new_centroid = []
    for i in range(new_data.shape[1]):  # calculating mean of assigned data columnwise for relocation
        if len(new_data[:, i]) != 0:
            new_centroid.append(np.mean(new_data[:, i]))
        else:
            new_centroid.append(0)
    return new_centroid


def init_centroids(k, data):
    n_indices = np.random.choice(range(data.shape[0]), size=k,
                                 replace=False)  # selecting random data point to initialize centroids randomly
    centroids = [data[x] for x in n_indices]
    centroids = np.array(centroids)
    return centroids


def assign_centroid(data, centroids):
    ncluster = centroids.shape[0]
    dist = np.matrix(np.ones((150, ncluster)) * np.inf)  # initializing array with infinity as values
    for i in range(0, ncluster):
        dist = np.append(dist, np.linalg.norm(data - centroids[i], axis=1).reshape(-1, 1),
                         axis=1)  # appending euclidean distance between data points an centroids to array
    dist_matrix = dist[:, ncluster:]  # slicing to remove infinity values
    assignment = np.argmin(dist, axis=1)  # assigning data points to closest centroid
    return assignment, dist_matrix


def number1():
    centroids, dist_matrix = kmeans(iris_data, 3)[:2]
    colors = np.array(['red', 'green', 'blue'])  # colors for clusters
    plt.title('3 Clusters, Sepal Length and Sepal width')  # plot of clusters and sepal length and sepal width
    plt.scatter(iris_data[:, 0], iris_data[:, 1], marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', color=colors, s=150)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.show()
    plt.title('3 Clusters, Sepal Length and Petal width')  # plot of clusters and sepal length and petal width
    plt.scatter(iris_data[:, 0], iris_data[:, 3], marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 3], marker='o', color=colors, s=150)
    plt.xlabel('Sepal length')
    plt.ylabel('Petal width')
    plt.show()


def number2():
    distance_lst = []  # initializing distance list to plot later
    for i in range(1, 11):
        centroids, dist_matrix = kmeans(iris_data, i)[:2]  # executing kmeans
        mean_distance = np.mean(dist_matrix.min(axis=1))  # calculating mean distance
        distance_lst.append(mean_distance)  # saving mean distance for every ncluster
    plt.title('Optimal Number of Clusters')  # mean cluster distance for ncluster 1-10
    plt.plot(list(range(1, 11)), distance_lst)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Mean Distance')
    plt.show()


if __name__ == '__main__':
    number1()
    number2()
