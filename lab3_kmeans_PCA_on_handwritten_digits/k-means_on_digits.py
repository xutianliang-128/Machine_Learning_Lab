from __future__ import print_function

import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import misc
from struct import unpack

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def plot_mean_image(X, log):
    meanrow = X.mean(0)
    #print(np.shape(X))
    # present the row vector as an image
    plt.figure(figsize=(3,3))
    plt.imshow(np.reshape(meanrow,(28,28)), cmap=plt.cm.binary)
    plt.title('Mean image of ' + log)
    plt.show()

def get_labeled_data(imagefile, labelfile):
    """
    Read input-vector (image) and target class (label, 0-9) and return it as list of tuples.
    Adapted from: https://martin-thoma.com/classify-mnist-with-pybrain/
    """
    # Open the images with gzip in read binary mode
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Read the binary data
    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    X = np.zeros((N, rows * cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros(N, dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        for id in range(rows * cols):
            tmp_pixel = images.read(1)  # Just a single byte
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            X[i][id] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (X, y)


def my_clustering_mnist(X, y, n_clusters):
    # =======================================
    #   1. Cluster images into n_clusters clusters using the k-means implemented by yourself or the one provided in scikit-learn.
    #
    #   2. Plot centers of clusters as images and combine these images in a single figure.
    #
    #   3. Return scores like this: return [score, score, score, score]
    ####################################################################
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    cen=kmeans.cluster_centers_
    cent=[]
    plt.figure(n_clusters)
    j=1
    for i in cen:
        plt.subplot(3,4,j)
        plt.imshow(i.reshape(28,28))
        j+=1
    plt.show()
    y_pred=kmeans.predict(X).T
    #print(y_pred)
    clustafter=y_pred.tolist()

    ari = metrics.adjusted_rand_score(y, clustafter)
    ami = metrics.adjusted_mutual_info_score(y, clustafter)
    vme = metrics.v_measure_score(y, clustafter)
    sil = metrics.silhouette_score(X, clustafter, metric='euclidean')

    return [ari,ami,vme,sil]

def main():
    # Load the dataset
    fname_img = 't10k-images.idx3-ubyte'
    fname_lbl = 't10k-labels.idx1-ubyte'
    [X, y]=get_labeled_data(fname_img, fname_lbl)

    # Plot the mean image
    plot_mean_image(X, 'all images')


    # Clustering
    range_n_clusters = [8, 9, 10, 11, 12]
    ari_score = [None] * len(range_n_clusters)
    mri_score = [None] * len(range_n_clusters)
    v_measure_score = [None] * len(range_n_clusters)
    silhouette_avg = [None] * len(range_n_clusters)

    for n_clusters in range_n_clusters:
        i = n_clusters - range_n_clusters[0]
        print("Number of clusters is: ", n_clusters)
        [ari_score[i], mri_score[i], v_measure_score[i], silhouette_avg[i]] = my_clustering_mnist(X, y, n_clusters)
        print('The ARI score is: ', ari_score[i])
        print('The MRI score is: ', mri_score[i])
        print('The v-measure score is: ', v_measure_score[i])
        print('The average silhouette score is: ', silhouette_avg[i])

    ####################################################################
    # you need to
    #   plot scores of all four evaluation metrics as functions of n_clusters in a single figure.
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
