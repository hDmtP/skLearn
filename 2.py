from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
# type(iris)
X = iris.data[:,2]
y = iris.target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=5)

X_ = X_train.reshape(-1,1)
X__ = X_test.reshape(-1,1)
y_ = y_train.reshape(-1,1)
y__ = y_test.reshape(-1,1)

model = svm.SVC(kernel='linear')
model.fit(X_,y_)

y_pred = model.predict(X__)

from sklearn.metrics import accuracy_score

print(accuracy_score(y__,y_pred))

######## KNN

print(X.shape)
print(y.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_,y_)

import numpy as np

a = np.array([4,5,6,7])
knn.predict([a])

# 35: 07 (try it on jupyteer notebook)