import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals.joblib import dump, load
# preprocess data
x_data = np.genfromtxt('../X_train', delimiter=',')[1:,]
y_data = np.genfromtxt('../Y_train', dtype=int)[1:,]
# 
#gb_clf = GradientBoostingClassifier()
gb_clf = GradientBoostingClassifier(n_estimators=500, max_leaf_nodes=16*2*2*2)
gb_clf.fit(x_data,y_data)
accuracy = gb_clf.score(x_data,y_data)
print(accuracy)
dump(gb_clf, str(accuracy)+'-gb.joblib')
