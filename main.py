import matplotlib.collections
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import csv


from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn import svm
import smogn
from scipy.optimize import minimize
from math import sqrt

np.set_printoptions(threshold=np.inf)

# White Wines
white_raw_data = pd.read_csv("clean_white.csv")
white_raw_data.head()

white_raw_data.isna().sum()

X = white_raw_data.drop(["quality"], axis=1)
y = white_raw_data["quality"]

y.plot.hist()
plt.show()

# Resample the minority class. You can change the strategy to 'auto' if you are not sure.
sm = SMOTE(sampling_strategy='auto', random_state=2, k_neighbors=2)

# Fit the model to generate the data.
oversampled_white_X, oversampled_white_Y = sm.fit_sample(X, y)
oversampled_white_data = pd.concat([pd.DataFrame(oversampled_white_X), pd.DataFrame(oversampled_white_Y)], axis=1)

oversampled_white_data.to_csv('oversampled_clean_data_white.csv', index=False)

oversampled_white_data['quality'].plot.hist()
plt.show()

mean = oversampled_white_X.mean()
std = oversampled_white_X.std()
normalized_white_X = (oversampled_white_X - mean) / std
normalized_white_X.head()

X_train, X_test, y_train, y_test = train_test_split(normalized_white_X, oversampled_white_Y, test_size=0.18,
                                                    random_state=42)

train_normalized_white_data = X_train.copy()
train_normalized_white_data['quality'] = y_train

test_normalized_white_data = X_test.copy()
test_normalized_white_data['quality'] = y_test
normalized_white_data = pd.concat([train_normalized_white_data, test_normalized_white_data])
normalized_white_data.to_csv('normalized_clean_data_white.csv', index=False)

sns.set(rc={'figure.figsize': (10, 8)})
corr = oversampled_white_data.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap="YlGnBu",
            annot=True,
            fmt=".2f")
plt.show()

# Red wines

red_data = pd.read_csv("clean_red.csv")
red_data.head()

X = red_data.drop(["quality"], axis=1)
y = red_data["quality"]

y.plot.hist()
plt.show()

# Resample the minority class. You can change the strategy to 'auto' if you are not sure.
sm = SMOTE(sampling_strategy='auto', random_state=2, k_neighbors=2)

# Fit the model to generate the data.
oversampled_red_X, oversampled_red_Y = sm.fit_sample(X, y)
oversampled_red_data = pd.concat([pd.DataFrame(oversampled_red_X), pd.DataFrame(oversampled_red_Y)], axis=1)

oversampled_red_data.to_csv('oversampled_clean_data_red.csv', index=False)

oversampled_red_data['quality'].plot.hist()
plt.show()

mean = oversampled_red_X.mean()
std = oversampled_red_X.std()
normalized_red_X = (oversampled_red_X - mean) / std
normalized_red_X.head()

X_train, X_test, y_train, y_test = train_test_split(normalized_red_X, oversampled_red_Y, test_size=0.18,
                                                    random_state=42)

train_normalized_red_data = X_train.copy()
train_normalized_red_data['quality'] = y_train

test_normalized_red_data = X_test.copy()
test_normalized_red_data['quality'] = y_test
normalized_red_data = pd.concat([train_normalized_red_data, test_normalized_red_data])
normalized_red_data.to_csv('normalized_clean_data_red.csv', index=False)

sns.set(rc={'figure.figsize': (10, 8)})
corr = oversampled_red_data.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap="YlGnBu",
            annot=True,
            fmt=".2f")
plt.show()

File_name = 'normalized_clean_data_white.csv'
# File_name = 'normalized_clean_data_red.csv'

data = pd.read_csv(File_name)
X = data.drop(['quality'], axis=1)
y = data['quality']
data.head()

data.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Question 1.a - Regrestion
# we'll try out three regresion models and compair them.
# 1. linear and polynomial regrestion
# 2. SVM Regresor
# 3. neural network regresor


# Linear Regrestion

reg = LinearRegression().fit(X_train, y_train)
print('r2 score = ', reg.score(X_test, y_test))

a = reg.predict(X_train)
train_rmse = (mean_squared_error(a, y_train)) ** 0.5
print('train_rmse =', train_rmse)
b = reg.predict(X_test)
test_rmse = (mean_squared_error(b, y_test)) ** 0.5
print('test_rmse =', test_rmse)

reg.coef_

# Polynomial Regression


p_reg = Pipeline([('poly', PolynomialFeatures(degree=2)),
                  ('linear', LinearRegression(fit_intercept=False))])

p_reg = p_reg.fit(X_train, y_train)
p_reg.named_steps['linear'].coef_

from sklearn.metrics import mean_squared_error

print('r2 score = ', p_reg.score(X_test, y_test))
a = p_reg.predict(X_train)
train_rmse = (mean_squared_error(a, y_train)) ** 0.5
print('train_rmse =', train_rmse)
b = p_reg.predict(X_test)
test_rmse = (mean_squared_error(b, y_test)) ** 0.5
print('test_rmse =', test_rmse)

# SVM Regresor


SVR_rbf = svm.SVR(kernel='rbf')
SVR_lin = svm.SVR(kernel='linear')
SVR_poly = svm.SVR(kernel='poly', degree=2)
SVR_rbf.fit(X_train, y_train)
SVR_lin.fit(X_train, y_train)
SVR_poly.fit(X_train, y_train)

print('SVR_rbf score = ', SVR_rbf.score(X_train, y_train))
a = SVR_rbf.predict(X_train)
train_rmse = (mean_squared_error(a, y_train)) ** 0.5
print('SVR_rbf train_rmse = ', train_rmse)
b = SVR_rbf.predict(X_test)
test_rmse = (mean_squared_error(b, y_test)) ** 0.5
print('SVR_rbf test_rmse = ', test_rmse, '\n')

print('SVR_lin score = ', SVR_lin.score(X_train, y_train))
a = SVR_lin.predict(X_train)
train_rmse = (mean_squared_error(a, y_train)) ** 0.5
print('SVR_lin train_rmse = ', train_rmse)
b = SVR_lin.predict(X_test)
test_rmse = (mean_squared_error(b, y_test)) ** 0.5
print('SVR_lin test_rmse = ', test_rmse, '\n')

print('SVR_poly score = ', SVR_poly.score(X_train, y_train))
a = SVR_poly.predict(X_train)
train_rmse = (mean_squared_error(a, y_train)) ** 0.5
print('SVR_poly train_rmse = ', train_rmse)
b = SVR_poly.predict(X_test)
test_rmse = (mean_squared_error(b, y_test)) ** 0.5
print('SVR_poly test_rmse = ', test_rmse)

# Deep Neural Network Regresor

import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.layers import Dropout
from keras.models import Model
import tensorflow as tf

DNN_model = Sequential()
DNN_model.add(Dense(100, input_dim=11, activation='selu'))
DNN_model.add(Dropout(0.1))
DNN_model.add(Dense(128, activation='selu'))
DNN_model.add(Dropout(0.1))
DNN_model.add(Dense(128, activation='selu'))
DNN_model.add(Dropout(0.1))
DNN_model.add(Dense(2, activation='selu'))
DNN_model.add(Dense(1))

DNN_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
history = DNN_model.fit(X_train, y_train, epochs=100, validation_split=0.3)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.figure(figsize=(12, 7))
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('mean_squared_error')
plt.ylabel('mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.figure(figsize=(12, 7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

a = DNN_model.predict(X_train)
train_rmse = (mean_squared_error(a, y_train)) ** 0.5
print('NN train_rmse = ', train_rmse)
b = DNN_model.predict(X_test)
test_rmse = (mean_squared_error(b, y_test)) ** 0.5
print('NN test_rmse = ', test_rmse)


# Clustering


# Dimension Reduction

def printPrejections(reducted_df: pd.DataFrame, model_name: str):
    for i in range(0, 2):
        for j in range(i + 1, 3):
            plotName = model_name + ' (dim ' + str(i) + '),(dim ' + str(j) + ')'
            dim1 = '(dim ' + str(i) + ')'
            dim2 = '(dim ' + str(j) + ')'
            vis = pd.DataFrame()
            vis[dim1] = reducted_df[i]
            vis[dim2] = reducted_df[j]
            vis['y'] = reducted_df['quality']
            plt.figure(figsize=(16, 10))
            sns.scatterplot(
                x=dim1, y=dim2,
                hue="y",
                palette=sns.color_palette("hls", len(set(list(reducted_df['quality'])))),
                data=vis,
                legend="full",
                alpha=0.8
            ).set_title(plotName)
            plt.show()


## pca
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(X)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)

pca_df = pd.DataFrame(pca.transform(X))
pca_df['quality'] = y

import seaborn as sns

sns.set(rc={'figure.figsize': (10, 8)})
corr = pca_df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap="YlGnBu",
            annot=True,
            fmt=".2f")
plt.show()

printPrejections(pca_df, 'PCA')

## Auto Encoder

input_dim = Input(shape=(11,))
# DEFINE THE DIMENSION OF ENCODER ASSUMED 3
encoding_dim = 3
# DEFINE THE ENCODER LAYERS
encoded1 = Dense(9, activation='selu')(input_dim)
encoded2 = Dense(6, activation='selu')(encoded1)
encoded3 = Dense(4, activation='selu')(encoded2)
encoded4 = Dense(encoding_dim, activation='relu')(encoded3)
# DEFINE THE DECODER LAYERS
decoded1 = Dense(4, activation='selu')(encoded4)
decoded2 = Dense(6, activation='selu')(decoded1)
decoded3 = Dense(9, activation='selu')(decoded2)
decoded4 = Dense(11, activation='sigmoid')(decoded3)
# COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
autoencoder = Model(input_dim, decoded4)
# CONFIGURE AND TRAIN THE AUTOENCODER
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs=100, batch_size=200, shuffle=True, validation_data=(X_test, X_test))
# THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
encoder = Model(input_dim, encoded4)
encoded_input = Input(shape=(encoding_dim,))
encoded_out = encoder.predict(X_test)

AE_data = pd.DataFrame(encoded_out)
AE_data['quality'] = y_test.values

sns.set(rc={'figure.figsize': (10, 8)})
corr = AE_data.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap="YlGnBu",
            annot=True,
            fmt=".2f")
plt.show()

printPrejections(AE_data, 'Auto Encoder')


## LLE

from sklearn.manifold import LocallyLinearEmbedding

embedding = LocallyLinearEmbedding(n_components=3)
X_transformed = embedding.fit_transform((X - X.mean())/X.std())
X_transformed.shape

LLE_data = pd.DataFrame(X_transformed)
LLE_data['quality'] = y.values
import seaborn as sns
sns.set(rc={'figure.figsize':(10,8)})
corr = LLE_data.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           cmap="YlGnBu",
            annot=True,
           fmt=".2f")
plt.show()

printPrejections(LLE_data, 'LLE')

## Linear Models Compereson

import math

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


red_file = 'normalized_clean_data_red.csv'
white_file = 'normalized_clean_data_white.csv'
red_data = pd.read_csv(red_file)
white_data = pd.read_csv(white_file)

all_data = pd.concat([red_data, white_data])
red_X_train, red_X_test, red_y_train, red_y_test = \
    train_test_split(red_data.drop(['quality'], axis=1), red_data['quality'], test_size=0.20, random_state=42)
white_X_train, white_X_test, white_y_train, white_y_test = \
    train_test_split(white_data.drop(['quality'], axis=1), white_data['quality'], test_size=0.20, random_state=42)
all_X_train, all_X_test, all_y_train, all_y_test = \
    train_test_split(all_data.drop(['quality'], axis=1), all_data['quality'], test_size=0.20, random_state=42)




# Linear Regrestion

red_lin_reg = LinearRegression().fit(red_X_train, red_y_train)
white_lin_reg = LinearRegression().fit(white_X_train, white_y_train)
all_lin_reg = LinearRegression().fit(all_X_train, all_y_train)

red_lin_reg_vec = np.array(red_lin_reg.coef_)
white_lin_reg_vec = np.array(white_lin_reg.coef_)
all_lin_reg_vec = np.array(all_lin_reg.coef_)


print(f'size of red_vec = {np.linalg.norm(red_lin_reg_vec)}')
print(f'size of white_vec = {np.linalg.norm(white_lin_reg_vec)}')
print(f'size of all_vec = {np.linalg.norm(all_lin_reg_vec)}')


print(f'angle between red and white = {angle(red_lin_reg_vec, white_lin_reg_vec)}')
print(f'angle between all and white = {angle(all_lin_reg_vec, white_lin_reg_vec)}')
print(f'angle between red and all = {angle(red_lin_reg_vec, all_lin_reg_vec)}')

#linear SVM


red_lin_SVM = svm.SVR(kernel='linear').fit(red_X_train, red_y_train)
white_lin_SVM = svm.SVR(kernel='linear').fit(white_X_train, white_y_train)
all_lin_SVM = svm.SVR(kernel='linear').fit(all_X_train, all_y_train)

red_lin_SVM_vec = np.array(red_lin_SVM.coef_[0])
white_lin_SVM_vec = np.array(white_lin_SVM.coef_[0])
all_lin_SVM_vec = np.array(all_lin_SVM.coef_[0])


print(f'size of red_vec = {np.linalg.norm(red_lin_SVM_vec)}')
print(f'size of white_vec = {np.linalg.norm(white_lin_SVM_vec)}')
print(f'size of all_vec = {np.linalg.norm(all_lin_SVM_vec)}')


print(f'angle between red and white = {angle(red_lin_SVM_vec, white_lin_SVM_vec)}')
print(f'angle between all and white = {angle(all_lin_SVM_vec, white_lin_SVM_vec)}')
print(f'angle between red and all = {angle(red_lin_SVM_vec, all_lin_SVM_vec)}')

#PCA
from sklearn.decomposition import PCA

red_pca = PCA(n_components=3)
white_pca = PCA(n_components=3)
all_pca = PCA(n_components=3)

red_pca.fit(red_data.drop(['quality'], axis=1))
white_pca.fit(white_data.drop(['quality'], axis=1))
all_pca.fit(all_data.drop(['quality'], axis=1))

print(red_pca.components_)
print(white_pca.components_)
print(all_pca.components_)

red_pca_vec_arr = [np.array(red_pca.components_[0]), np.array(red_pca.components_[1]), np.array(red_pca.components_[2])]
white_pca_vec_arr = [np.array(white_pca.components_[0]), np.array(white_pca.components_[1]), np.array(white_pca.components_[2])]
all_pca_vec_arr = [np.array(all_pca.components_[0]), np.array(all_pca.components_[1]), np.array(all_pca.components_[2])]

for v in red_pca_vec_arr:
    print(f'size of red_vec = {np.linalg.norm(v)}')
for v in white_pca_vec_arr:
    print(f'size of white_vec = {np.linalg.norm(white_pca_vec_arr)}')
for v in all_pca_vec_arr:
    print(f'size of all_vec = {np.linalg.norm(all_pca_vec_arr)}')

for i in range(2):
    print(f'{i} angle between red and white = {angle(red_pca_vec_arr[i], white_pca_vec_arr[i])}')
    print(f'{i} angle between all and white = {angle(all_pca_vec_arr[i], white_pca_vec_arr[i])}')
    print(f'{i} angle between red and all = {angle(red_pca_vec_arr[i], all_pca_vec_arr[i])}')


# question number 3


# World's greatest grape
def CustumLossFunction(x: list, model):
    X_df = pd.DataFrame([x], columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                                                    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
                                                    'density', 'pH', 'sulphates', 'alcohol'])
    prediction_value = model.predict(X_df)
    while type(prediction_value) != int:
        prediction_value = prediction_value[0]
    return abs(10-prediction_value)

methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr']

model = SVR_rbf
min_error = 100
best_method = ''
x0 = list(data[data.quality == 9].mean().to_frame().T.drop(['quality'],axis=1).values[0])
for m in methods:
    res = minimize(CustumLossFunction, x0, method=m, tol=1e-6, options={'maxiter': 10000})
    if res.fun < min_error:
        best_method = m
        min_error = res.fun

print(best_method)
print(min_error)
res = minimize(CustumLossFunction, x0, method=best_method, tol=1e-6, options={'maxiter': 10000})
print('best grape SVR_rbf score: ', SVR_rbf.predict(pd.Series(res.x).to_frame().T)[0])
print('best grape DNN score: ', DNN_model.predict(pd.Series(res.x).to_frame().T)[0][0])
print('best grape SVR_lin score: ', SVR_lin.predict(pd.Series(res.x).to_frame().T)[0])
print('best grape SVR_poly score: ', SVR_poly.predict(pd.Series(res.x).to_frame().T)[0])
print('best grape Linear regression score: ', reg.predict(pd.Series(res.x).to_frame().T)[0])


