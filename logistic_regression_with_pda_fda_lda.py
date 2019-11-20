import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Turn down for faster convergence
t0 = time.time()

# Load data from https://www.openml.org/d/554
mnist = fetch_openml('mnist_784', version=1)

train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train_img)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

# Make an instance of the Model
pca = PCA()
pca.fit(train_img)

# Available solvers => svd, lsqr, eigen
lda = LDA()
lda.fit(train_img, train_lbl)

#PCA
train_img_pca = pca.transform(train_img)
test_img_pca = pca.transform(test_img)

#LDA
train_img_lda = lda.transform(train_img)
test_img_lda = lda.transform(test_img)

# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegrPca = LogisticRegression(solver = 'lbfgs', max_iter=1000, multi_class='auto')
logisticRegrPca.fit(train_img_pca, train_lbl)

scorePca = logisticRegrPca.score(test_img_pca, test_lbl)

logisticRegrLda = LogisticRegression(solver = 'lbfgs', max_iter=1000, multi_class='auto')
logisticRegrLda.fit(train_img_lda, train_lbl)

scoreLda = logisticRegrLda.score(test_img_lda, test_lbl)

logisticRegrSalt = LogisticRegression(solver = 'lbfgs', max_iter=1000, multi_class='auto')
logisticRegrSalt.fit(train_img, train_lbl)

scoreSalt = logisticRegrSalt.score(test_img, test_lbl)

y_pred_salt = logisticRegrSalt.predict(test_img)
cm_salt = confusion_matrix(test_lbl, y_pred_salt)
f1_salt = f1_score(test_lbl, y_pred_salt, average="macro")
precision_salt = precision_score(test_lbl, y_pred_salt, average="macro")
recall_salt = recall_score(test_lbl, y_pred_salt, average="macro")
print("Salt Score : ", scoreSalt)
print("Salt F Measure Score : ", f1_salt)
print("Salt Precision Score : ", precision_salt)
print("Salt Recall Score : ", recall_salt)
print("Salt Confusion Matrix : ")
print(cm_salt)

y_pred_pca = logisticRegrPca.predict(test_img_pca)
cm_pca = confusion_matrix(test_lbl, y_pred_pca)
f1_pca = f1_score(test_lbl, y_pred_pca, average="macro")
precision_pca = precision_score(test_lbl, y_pred_pca, average="macro")
recall_pca = recall_score(test_lbl, y_pred_pca, average="macro")
print("PCA Score : ", scorePca)
print("PCA F MEASURE : ", f1_pca)
print("PCA Precision Score : ", precision_pca)
print("PCA Recall Score : ", recall_pca)
print("PCA Confusion Matrix : ")
print(cm_pca)

y_pred_lda = logisticRegrLda.predict(test_img_lda)
cm_lda = confusion_matrix(test_lbl, y_pred_lda)
f1_lda = f1_score(test_lbl, y_pred_lda, average="macro")
precision_lda = precision_score(test_lbl, y_pred_lda, average="macro")
recall_lda = recall_score(test_lbl, y_pred_lda, average="macro")
print("LDA Score : ", scoreLda)
print("LDA F MEASURE : ", f1_lda)
print("LDA Precision Score : ", precision_lda)
print("LDA Recall Score : ", recall_lda)
print("LDA Confusion Matrix : ")
print(cm_lda)

run_time = time.time() - t0
print('Example run in %.3f s' % run_time)
