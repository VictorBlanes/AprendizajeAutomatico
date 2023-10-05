import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from Adaline import Adaline
from sklearn.svm import SVC

# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=2,
                           random_state=9)
y[y == 0] = -1

# Els dos algorismes es beneficien d'estandaritzar les dades
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X)

# Entrenam un perceptron
perceptron = Adaline(eta=0.0005, n_iter=60)
perceptron.fit(X_transformed, y)
y_prediction = perceptron.predict(X)

# Entrenam una SVM linear (classe SVC)

svc = SVC(C=1.0, kernel='linear')
svc.fit(X_transformed, y)

interecept = svc.intercept_
weights = svc.coef_
support_vectors = svc.support_vectors_

plt.figure(1)

#  Mostram els resultats Adaline
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
m = -perceptron.w_[1] / perceptron.w_[2]
origen = (0, -perceptron.w_[0] / perceptron.w_[2])
plt.axline(xy1=origen, slope=m, c="blue", label="Adaline")

#  Mostram els resultats SVM
slope_svc = -weights[0][0] / weights[0][1]
origen_svc = (0, -interecept[0] / weights[0][1])
plt.axline(xy1=origen_svc, slope=slope_svc, c="green", label="SVM")
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors="none", edgecolors="red")

plt.legend()
plt.show()
