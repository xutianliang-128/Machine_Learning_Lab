import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
import seaborn
seaborn.set()

df = pd.read_csv('imports-85.data',
                 header=None,
                 names = ["symboling","normalized.losses","make","fuel.type",
         "aspiration","num.of.doors","body.style","drive.wheels",
         "engine.location","wheel.base","length","width",
         "height","curb.weight","engine.type","num.of.cylinders",
         "engine.size","fuel.system","bore","stroke",
         "compression.ratio","horsepower","peak.rpm","city.mpg",
         "highway.mpg","price"],
                 na_values=('?'))

df=df.dropna()

col_n = ['city.mpg','horsepower','engine.size', 'peak.rpm','price']

X_choose_train = pd.DataFrame(df,columns=col_n)

X_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_choose_train)
X_final_train = X_train_scaled[:,0:4]

X_matt=np.mat(X_final_train)
one = np.ones((X_matt.shape[0], 1))
X_mat=np.column_stack((one,X_matt))

train_price=X_train_scaled[:,4].reshape(-1,1)
y=train_price

theta=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X_mat),X_mat)),np.transpose(X_mat)),y)

print('Parameter theta calculated by normal equation: \n',theta)

#############################################

model = SGDRegressor()
model.fit(X_matt,y.ravel())
print('Parameter theta calculated by SDG: ',model.intercept_, model.coef_)

# train_cm=X_train_scaled[:,0].reshape(-1,1)
# train_hp=X_train_scaled[:,1].reshape(-1,1)
# train_es=X_train_scaled[:,2].reshape(-1,1)
# train_pr=X_train_scaled[:,3].reshape(-1,1)
train_price=X_train_scaled[:,4].reshape(-1,1)
