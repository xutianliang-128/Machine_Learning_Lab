import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split




n_samples = 10000

centers = [(-1, 1), (1, -1), (1, 1), (-1,-1)]
X, y = make_blobs(n_samples=n_samples, n_features=4, cluster_std=1.8,
                  centers=centers, shuffle=False, random_state=19)

y[:n_samples // 4] = 0
y[n_samples // 4: n_samples // 4*2] = 1
y[n_samples // 4 * 2: n_samples // 4*3] = 2
y[n_samples // 4*3: n_samples // 4*4] = 3

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=19)
log_reg = linear_model.LogisticRegression()


log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred)
plt.title('Classification with Logistic Regression')
plt.show()

# plt.scatter(X_test[:,0], X_test[:,1], c=y_test)
# plt.show()

wrong = np.count_nonzero(y_test - y_pred)
print('Number of wrong predictions is: ' + str(wrong))
