import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

X_train,X_test=train_test_split(df,test_size=0.2, random_state=1)

col_n = ['price', 'horsepower']

X_choose_train = pd.DataFrame(X_train, columns=col_n)
X_choose_test = pd.DataFrame(X_test, columns=col_n)

X_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_choose_train)
X_test_scaled = X_scaler.transform(X_choose_test)
train_price=X_train_scaled[:,0].reshape(-1,1)
train_hp=X_train_scaled[:,1].reshape(-1,1)
test_price=X_test_scaled[:,0].reshape(-1,1)
test_hp=X_test_scaled[:,1].reshape(-1,1)


model =linear_model.LinearRegression()

model.fit(train_hp,train_hp)


plt.scatter(test_hp,test_price,c='b')
plt.scatter(test_hp,test_hp*model.coef_+model.intercept_,c='r',marker='v')
plt.title('Linear regression on cleaned and standardized test data')
plt.xlabel("Standardized horsepower")
plt.ylabel("Standardized price")
plt.show()

# plt.plot(range(1,len(d)+1),d)


