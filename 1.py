from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[1,2,3],[11,12,13]]
y = [0,1]
clf.fit(X,y)
print(clf.predict(X))