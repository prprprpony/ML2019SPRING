import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import dump, load
# preprocess data
x_data = np.genfromtxt('../X_train', delimiter=',')[1:,]
y_data = np.genfromtxt('../Y_train', dtype=int)[1:,]
# 
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16*2*2*2, n_jobs=-1)
rnd_clf.fit(x_data,y_data)
accuracy = np.mean(rnd_clf.predict(x_data) == y_data)
print(accuracy)
dump(rnd_clf, str(accuracy)+'-rnd.joblib')
