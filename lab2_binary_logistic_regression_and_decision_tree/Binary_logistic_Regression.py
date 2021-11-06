import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler


def logistic_func(x):

    # Output: logistic(x)
    ####################################################################
    L = 1 / (1 + np.exp(-x))
    return L

################################################################





def train(X_train, y_train, tol=10 ** -4):

    # Output: the weight update result [w_0, w_1, w_2, ...]
    ####################################################################

    LearningRate = 0.05
    r = LearningRate
    feature_num = len(X_train[0])

    xx=X_train
    yy=y_train

    def L(x):
        l = logistic_func(x)
        return l

    def g(x, ww, w0):
        g_result = w0 + np.dot(ww, x)
        return g_result

    w0=float(0)
    ww = [0] * feature_num

    def algo(xx, yy, ww, w0):
        sum1=0
        for d in range(len(xx)):
            sum1 += yy[d]-L(g(xx[d],ww,w0))
        w0_next = w0 + r*sum1
        sum2=0

        ww_next=ww

        for i in range(len(xx[0])):
            sumd = 0
            for d in range(len(xx)):
                sumd += xx[d][i]*(yy[d] - L(g(xx[d],ww,w0)))
            ww_next[i] = ww_next[i] + sumd*r

        return ww_next, w0_next

    evalu = 1000

    while True:
        w_together_now = [w0]+ww
        ww_next, w0_next = algo(xx, yy, ww, w0)
        w_together_next = [w0_next]+ww_next

        w_diff=[(w_together_next[i]-w_together_now[i]) for i in range(len(w_together_next))]
        evalu = sum(np.abs(np.array(w_diff)))
        ww , w0 = ww_next , w0_next
        if evalu < tol:
            weights = w_together_next
            break

    return weights


#
#
def train_matrix(X_train, y_train, tol=10 ** -4):
#

#

    # Output: the weight update result [w_0, w_1, w_2, ...]
    ####################################################################

    LearningRate = 0.04
    r=LearningRate

    def sigmoid(x_arr):
        return 1.0 / (1 + np.exp(-x_arr))

    def logistic_regression(x_arr, y_arr, r):
        x_mat = np.mat(x_arr)
        y_mat = np.mat(y_arr)
        weight = np.ones((x_mat.shape[1], 1))
        while True:
            h = sigmoid(x_mat * weight)
            error = (y_mat - h)
            weight_next = weight + r * x_mat.T * error
            if sum(np.abs(weight_next-weight)) < tol:
                return weight_next
            else:
                weight = weight_next

    data_mat = np.asmatrix(X_train)
    length = len(X_train)
    ones = np.ones((length,1))
    data_mat = np.hstack((ones,data_mat))
    label_mat = np.asmatrix(y_train)
    weights = logistic_regression(data_mat, label_mat.T,r)

    return weights
#
def predict(X_test, weights):

    # The predict labels of all points in test dataset.
    ####################################################################


    #####################################

    xx = X_test
    yy = y_test

    def sigmoid(x_arr):
        return 1.0 / (1 + np.exp(-x_arr))

    data_mat = np.asmatrix(X_test)
    length = len(X_test)
    ones = np.ones((length, 1))
    data_mat = np.hstack((ones, data_mat))
    label_mat = np.asmatrix(y_test)
    x_mat = np.mat(data_mat)
    h = sigmoid(x_mat * weights)
    predictions=[]
    for i in h:
        if i >=0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    return np.array(predictions)


def plot_prediction(X_test, X_test_prediction):
    X_test1 = X_test[X_test_prediction == 0, :]
    X_test2 = X_test[X_test_prediction == 1, :]
    plt.scatter(X_test1[:, 0], X_test1[:, 1], color='red')
    plt.scatter(X_test2[:, 0], X_test2[:, 1], color='blue')
    plt.show()


# Data Generation
n_samples = 1000

centers = [(-1, -1), (5, 10)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.8,
                  centers=centers, shuffle=False, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# Experiments
# If you want to get predictions for train, run this code below

# ww = train(X_train, y_train)
# w=np.array([ww]).T
# X_test_prediction = predict(X_test, w)
# plot_prediction(X_test, X_test_prediction)
# plot_prediction(X_test, y_test)
#
# wrong = np.count_nonzero(y_test - X_test_prediction)
# print('Number of wrong predictions is: ' + str(wrong))

############################################################

# Experiments
# If you want to get predictions for train_matrix, run this code below

w = train_matrix(X_train, y_train)
X_test_prediction = predict(X_test, w)
plot_prediction(X_test, X_test_prediction)
plot_prediction(X_test, y_test)

wrong = np.count_nonzero(y_test - X_test_prediction)
print('Number of wrong predictions is: ' + str(wrong))
