import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def create_data(x1, x2, x3):
    x4 = -4.0 * x1
    x5 = 10 * x1 + 10
    x6 = -1 * x2 / 2
    x7 = np.multiply(x2, x2)
    x8 = -1 * x3 / 10
    x9 = 2.0 * x3 + 2.0
    X = np.hstack((x1, x2, x3, x4, x5, x6, x7, x8, x9))
    return X

def pca(X):
    '''
    PCA step by step
      1. normalize matrix X
      2. compute the covariance matrix of the normalized matrix X
      3. do the eigenvalue decomposition on the covariance matrix
    Return: [d, V]
      d is the column vector containing all the corresponding eigenvalues,
      V is the matrix containing all the eigenvectors.
    '''

    for i in range(X.shape[1]):
        X[:,i]=X[:,i]-sum(X[:,i])/len(X[:,i])#没有小数了
    c=np.cov(X,rowvar=0)
    [eigv ,eigvect]=np.linalg.eig(c)
    sortInd=np.argsort(-eigv)
    d=eigv[sortInd]
    V=eigvect[:,sortInd]

    # here d is the column vector containing all the corresponding eigenvalues.
    # V is the matrix containing all the eigenvectors,
    return [d, V]

def plot_figs(X):
    """
    1. perform PCA (you can use pca(X) completed by yourself) on this matrix X
    2. plot (a) The figure of eigenvalues v.s. the order of eigenvalues. All eigenvalues in a decreasing order.
    3. plot (b) The figure of POV v.s. the order of the eigenvalues.
    """
    # pca = PCA()
    # pca.fit(X)
    [d, _]=pca(X)
    fig=plt.figure(1)
    plt.plot(range(1,len(d)+1),d)

    fig=plt.figure(2)
    s=sum(d)
    a=[]
    for i in d:
        if len(a)!=0:
            a.append(a[-1]+i/s)
        else:
            a.append(i/s)
    plt.plot(range(1,len(a)+1),a)

    plt.show()
    return

def main():
    N = 1000
    shape = (N, 1)
    x1 = np.random.normal(0, 1, shape)  # samples from normal distribution
    x2 = np.random.exponential(10.0, shape)  # samples from exponential distribution
    x3 = np.random.uniform(-100, 100, shape)  # uniformly sampled data points
    X = create_data(x1, x2, x3)

    print(pca(X))
    plot_figs(X)

if __name__ == '__main__':
    main()

