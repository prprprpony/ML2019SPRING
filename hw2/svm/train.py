import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC
from sklearn.externals.joblib import dump, load
# preprocess data
x_data = np.genfromtxt('../X_train', delimiter=',')[1:,]
y_data = np.genfromtxt('../Y_train', dtype=int)[1:,]
# 
svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=0.125,loss="squared_hinge")),
        ))
svm_clf.fit(x_data,y_data)
accuracy = np.mean(svm_clf.predict(x_data) == y_data)
print(accuracy)
dump(svm_clf, str(accuracy)+'-svm.joblib')
