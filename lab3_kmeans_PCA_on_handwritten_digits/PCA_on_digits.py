import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.decomposition import PCA
import imageio

def load_data(digits, num):
    '''
    Loads all of the images into a data-array.

    The training data has 5000 images per digit,
    but loading that many images from the disk may take a while.  So, you can
    just use a subset of them, say 300 for training (otherwise it will take a
    long time to complete.

    Note that each image as a 28x28 grayscale image, loaded as an array and
    then reshaped into a single row-vector.

    Use the function display(row-vector) to visualize an image.

    '''
    totalsize = 0
    for digit in digits:
        totalsize += min([len(next(os.walk('train%d' % digit))[2]), num])
    print('We will load %d images' % totalsize)
    X = np.zeros((totalsize, 784), dtype = np.uint8)   #784=28*28
    for index in range(0, len(digits)):
        digit = digits[index]
        print('\nReading images of digit %d' % digit)
        for i in range(num):
            pth = os.path.join('train%d' % digit,'%05d.pgm' % i)
            image = imageio.imread(pth).reshape((1,784))
            # image = misc.imread(pth).reshape((1, 784))
            X[i + index * num, :] = image
        print('\n')
    return X

def plot_mean_image(X, digits = [0]):
    ''' example on presenting vector as an image
    '''
    plt.close('all')
    meanrow = X.mean(0)
    # present the row vector as an image
    plt.imshow(np.reshape(meanrow,(28,28)))
    plt.title('Mean image of digit ' + str(digits))
    plt.gray(), plt.xticks(()), plt.yticks(()), plt.show()

def main():
    digits = [0, 1, 2]
    # load handwritten images of digit 0, 1, 2 into a matrix X
    # for each digit, we just use 300 images
    # each row of matrix X represents an image
    X = load_data(digits, 300)
    # plot the mean image of these images!
    # plot_mean_image(X, digits)

    ####################################################################
    #   1. do the PCA on matrix X;
    #
    #   2. plot the eigenimages (reshape the vector to 28*28 matrix then use
    #   the function ``imshow'' in pyplot), save the images of eigenvectors
    #   which correspond to largest 9 eigenvalues. Save them in a single file
    #   ``eigenimages.jpg''.
    #
    #   3. plot the POV (the Portion of variance explained v.s. the number of
    #   components we retain), save the figure in file ``pov.jpg''
    ####################################################################

    pca=PCA(n_components=9)
    pca.fit(X)
    eigvect=[]
    for i in range(len(pca.components_)):
        j=np.array(pca.components_[i]).reshape(28,28)
        eigvect.append(j)
    fig=plt.figure()
    k=0
    plt.gray()
    for i in eigvect:
        k+=1
        #plt.title('The ' + str(k)+ "th image")
        plt.subplot(330+k)
        plt.imshow(i)
    plt.show()

    pca1 = PCA()
    pca1.fit(X)
    d=pca1.explained_variance_ratio_
    a=[]
    for i in range(len(d)):
        a.append(sum(d[:i]))
        if sum(d[:i]) >= 0.9:
            print(i)
    plt.plot(range(1,len(a)+1),a)

    plt.show()
if __name__ == '__main__':
    main()
