# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:37:11 2023

@author: Esterlita
"""

import os
import fnmatch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#buat fungsi untuk data x dan y
def make_train_data(uang, DIR):
    for img in os.listdir(DIR):
        if fnmatch.fnmatch(img, '*.jpg'):
            label = uang
            path = os.path.join(DIR, img)
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (150, 150))
            X.append(img)
            y.append(label)

#inisialisasikan x dan y
X = []
y = []

#membaca dataset
make_train_data('uang asli', 'asli')

#memeriksa jumlah dari data
print(len(X))
print(len(y))

#melakukan plot pada data
fig, ax = plt.subplots(3, 2)
fig.set_size_inches(12, 12)
for i in range(3):
    for j in range(2):
        l = np.random.randint(0, len(y))
        ax[i, j].imshow(X[l], cmap='gray')
        ax[i, j].set_title('Batik-' + y[l])
plt.tight_layout()

#mengatur LBP
radius = 2
n_points = 8 * radius
METHOD = 'uniform'

#menghitung feature LBP
listLBP = [local_binary_pattern(img, n_points, radius, METHOD) for img in X]
lbpHist = [np.histogram(lbp.ravel(), 256, [0, 256])[0] for lbp in listLBP]

#memperoleh hasil dari matriks x
X = np.array(lbpHist)

#mengaplikasikan LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#membagi data untuk di training dan di tes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#menginisialisasi model
model_svc = SVC(C=0.1, gamma=1, kernel='poly')
model_mlp = MLPClassifier(hidden_layer_sizes=(100, 15), alpha=0.05, max_iter=1000, activation='tanh')
model_knn = KNeighborsClassifier(n_neighbors=3)

#melakukan training pada model
model_svc.fit(X_train, y_train)
model_mlp.fit(X_train, y_train)
model_knn.fit(X_train, y_train)

#memperoleh prediksi label untuk training dan testing data
y_train_pred_svc = model_svc.predict(X_train)
y_test_pred_svc = model_svc.predict(X_test)
y_train_pred_mlp = model_mlp.predict(X_train)
y_test_pred_mlp = model_mlp.predict(X_test)
y_train_pred_knn = model_knn.predict(X_train)
y_test_pred_knn = model_knn.predict(X_test)

#melakukan plotting pada scatter plotnya untuk melakukan training dan testing
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
plt.title('Training Data - True Labels')
plt.subplot(2, 3, 2)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred_svc, cmap='viridis')
plt.title('Training Data - Predicted Labels (SVC)')
plt.subplot(2, 3, 3)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred_mlp, cmap='viridis')
plt.title('Training Data - Predicted Labels (MLP)')
plt.subplot(2, 3, 4)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
plt.title('Testing Data - True Labels')
plt.subplot(2, 3, 5)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred_svc, cmap='viridis')
plt.title('Testing Data - Predicted Labels (SVC)')
plt.subplot(2, 3, 6)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred_mlp, cmap='viridis')
plt.title('Testing Data - Predicted Labels (MLP)')
plt.tight_layout()

#menghitung dan menampilkan akurasi dan juga nilai knn
accuracy_svc_train = accuracy_score(y_train, y_train_pred_svc)
accuracy_svc_test = accuracy_score(y_test, y_test_pred_svc)
accuracy_mlp_train = accuracy_score(y_train, y_train_pred_mlp)
accuracy_mlp_test = accuracy_score(y_test, y_test_pred_mlp)
accuracy_knn_train = accuracy_score(y_train, y_train_pred_knn)
accuracy_knn_test = accuracy_score(y_test, y_test_pred_knn)
k_knn = model_knn.n_neighbors

print('\nAccuracy - Training (SVC)\t:', accuracy_svc_train)
print('Accuracy - Testing (SVC)\t:', accuracy_svc_test)
print('\nAccuracy - Training (MLP)\t:', accuracy_mlp_train)
print('Accuracy - Testing (MLP)\t:', accuracy_mlp_test)
print('\nAccuracy - Training (KNN)\t:', accuracy_knn_train)
print('Accuracy - Testing (KNN)\t:', accuracy_knn_test)
print('K value (KNN):', k_knn)

plt.show()