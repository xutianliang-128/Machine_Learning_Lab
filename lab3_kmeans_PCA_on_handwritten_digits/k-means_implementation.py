from __future__ import print_function
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics

def create_dataset():
    # Generate sample points
    centers = [[3,5], [5,1], [8,2], [6,8], [9,7]]
    X, y = make_blobs(n_samples=1000,centers=centers,cluster_std=[0.5, 0.5, 1, 1, 1],random_state=3320)
    ####################################################################
    # you need to
    #   1. Plot the data points in a scatter plot.
    #   2. Use color to represents the clusters.
    ####################################################################
    xx=[]
    yy=[]
    for i in X:
        xx.append(i[0])
        yy.append(i[1])
    plt.scatter(xx,yy,c=y)
    plt.show()
    return [X, y]

def my_clustering(X, y, n_clusters):
    # =======================================
    # you need to
    #   1. Implement the k-means by yourself
    #   and cluster samples into n_clusters clusters using your own k-means
    #
    #   2. Print out all cluster centers and sizes.
    #
    #   3. Plot all clusters formed,
    #   and use different colors to represent clusters defined by k-means.
    #   Draw a marker (e.g., a circle or the cluster id) at each cluster center.
    #
    #   4. Return scores like this: return [score, score, score, score]
    ####################################################################
    def dist(a, b, ax=1):
        return np.linalg.norm(a - b, axis=ax)

    def randCent(dataSet, k):
        n = np.shape(dataSet)[1]
        centroids = np.mat(np.zeros((k, n)))
        for j in range(n):
            minJ = min(dataSet[:, j])
            maxJ = max(dataSet[:, j])
            rangeJ = float(maxJ - minJ)
            centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
        return centroids

    def kMeans(dataSet, k, distMeans=dist, createCent=randCent):
        m = np.shape(dataSet)[0]
        clusterAssment = np.mat(np.zeros((m, 2)))  # store the sample belonging and the distance
        # clusterAssment
        centroids = createCent(dataSet, k)
        clusterChanged = True  # identify if the cluster has already converged.
        while clusterChanged:
            clusterChanged = False;
            for i in range(m):  # change the point belonging.
                minDist = np.inf;
                minIndex = -1;
                for j in range(k):
                    distJI = distMeans(centroids[j, :], dataSet[i, :])
                    if distJI < minDist:
                        minDist = distJI;
                        minIndex = j  # i belongs to j if it is closer to j.
                if clusterAssment[i, 0] != minIndex: clusterChanged = True;  # iterate if changed
                clusterAssment[i, :] = minIndex, minDist ** 2  # update the dictionary
            # print(centroids)
            for cent in range(k):  # recompute the middle point
                ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
                centroids[cent, :] = np.mean(ptsInClust, axis=0)  # compute the middle point of these points
        return centroids, clusterAssment



    myCentroids, clustAssing = kMeans(X, n_clusters)  # draw the graph
    kk = clustAssing.tolist()
    clustafter = []
    for k in kk:
        clustafter.append(k[0])
    clustafter = list(map(int, clustafter))
    xxx = []
    yyy = []
    pp = myCentroids.tolist()
    for i in pp:
        xxx.append(i[0])
        yyy.append(i[1])
    xx = []
    yy = []
    for i in X:
        xx.append(i[0])
        yy.append(i[1])
    plt.scatter(xx, yy, c=clustafter)
    plt.scatter(xxx, yyy, marker='*', c='r', s=50)
    plt.show()

    ari = metrics.adjusted_rand_score(y, clustafter)
    ami = metrics.adjusted_mutual_info_score(y, clustafter)
    vme = metrics.v_measure_score(y, clustafter)
    sil = metrics.silhouette_score(X, clustafter,metric='euclidean')

    return [ari,ami,vme,sil]

def main():
    X, y = create_dataset()
    my_clustering(X, y, 5)

    range_n_clusters = [2, 3, 4, 5, 6, 7]
    ari_score = [None] * len(range_n_clusters)
    mri_score = [None] * len(range_n_clusters)
    v_measure_score = [None] * len(range_n_clusters)
    silhouette_avg = [None] * len(range_n_clusters)

    for n_clusters in range_n_clusters:
        i = n_clusters - range_n_clusters[0]
        print("Number of clusters is: ", n_clusters)
        [ari_score[i], mri_score[i], v_measure_score[i], silhouette_avg[i]] = my_clustering(X, y, n_clusters)
        print('The ARI score is: ', ari_score[i])
        print('The MRI score is: ', mri_score[i])
        print('The v-measure score is: ', v_measure_score[i])
        print('The average silhouette score is: ', silhouette_avg[i])

    ####################################################################
    # Plot scores of all four evaluation metrics as functions of n_clusters in a single figure.
    ####################################################################

    fig, ax = plt.subplots()
    ax.plot(range_n_clusters, ari_score, label='ari')
    ax.plot(range_n_clusters, mri_score, label='mri')
    ax.plot(range_n_clusters, v_measure_score, label='v_score')
    ax.plot(range_n_clusters, silhouette_avg, label='silhouette')
    ax.set_title('scores_function')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()

