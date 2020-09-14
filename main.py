# %% [code]
from sklearn.preprocessing import PolynomialFeatures
# from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
# import smogn
from scipy.optimize import minimize
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn import svm
import numpy as np
import seaborn as sns
from scipy.stats import norm
import random
import math
import matplotlib.collections
from itertools import combinations
import networkx as nx

np.set_printoptions(threshold=np.inf)
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

np.set_printoptions(threshold=np.inf)

# %% [code]

# # White Wines
# white_raw_data = pd.read_csv("clean_white.csv")
# white_raw_data.head()
#
# white_raw_data.isna().sum()
#
# X = white_raw_data.drop(["quality"], axis=1)
# Y = white_raw_data["quality"]
#
# Y.plot.hist()
# plt.show()
#
# # Resample the minority class. You can change the strategy to 'auto' if you are not sure.
# sm = SMOTE(sampling_strategy='auto', random_state=2, k_neighbors=2)
#
# # Fit the model to generate the data.
# oversampled_white_X, oversampled_white_Y = sm.fit_sample(X, Y)
# oversampled_white_data = pd.concat([pd.DataFrame(oversampled_white_X), pd.DataFrame(oversampled_white_Y)], axis=1)
#
# oversampled_white_data.to_csv('oversampled_clean_data_white.csv', index=False)
#
# oversampled_white_data['quality'].plot.hist()
# plt.show()
#
# mean = oversampled_white_X.mean()
# std = oversampled_white_X.std()
# normalized_white_X = (oversampled_white_X - mean) / std
# normalized_white_X.head()
#
# X_train, X_test, Y_train, Y_test = train_test_split(normalized_white_X, oversampled_white_Y, test_size=0.18,
#                                                     random_state=42)
#
# train_normalized_white_data = X_train.copy()
# train_normalized_white_data['quality'] = Y_train
#
# test_normalized_white_data = X_test.copy()
# test_normalized_white_data['quality'] = Y_test
# normalized_white_data = pd.concat([train_normalized_white_data, test_normalized_white_data])
# normalized_white_data.to_csv('normalized_clean_data_white.csv', index=False)
#
# sns.set(rc={'figure.figsize': (10, 8)})
# corr = oversampled_white_data.corr()
# sns.heatmap(corr,
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values,
#             cmap="YlGnBu",
#             annot=True,
#             fmt=".2f")
# plt.show()
#
# # Red wines
#
# red_data = pd.read_csv("clean_red.csv")
# red_data.head()
#
# X = red_data.drop(["quality"], axis=1)
# Y = red_data["quality"]
#
# Y.plot.hist()
# plt.show()
#
# # Resample the minority class. You can change the strategy to 'auto' if you are not sure.
# sm = SMOTE(sampling_strategy='auto', random_state=2, k_neighbors=2)
#
# # Fit the model to generate the data.
# oversampled_red_X, oversampled_red_Y = sm.fit_sample(X, Y)
# oversampled_red_data = pd.concat([pd.DataFrame(oversampled_red_X), pd.DataFrame(oversampled_red_Y)], axis=1)
#
# oversampled_red_data.to_csv('oversampled_clean_data_red.csv', index=False)
#
# oversampled_red_data['quality'].plot.hist()
# plt.show()
#
# mean = oversampled_red_X.mean()
# std = oversampled_red_X.std()
# normalized_red_X = (oversampled_red_X - mean) / std
# normalized_red_X.head()
#
# X_train, X_test, Y_train, Y_test = train_test_split(normalized_red_X, oversampled_red_Y, test_size=0.18,
#                                                     random_state=42)
#
# train_normalized_red_data = X_train.copy()
# train_normalized_red_data['quality'] = Y_train
#
# test_normalized_red_data = X_test.copy()
# test_normalized_red_data['quality'] = Y_test
# normalized_red_data = pd.concat([train_normalized_red_data, test_normalized_red_data])
# normalized_red_data.to_csv('normalized_clean_data_red.csv', index=False)
#
# sns.set(rc={'figure.figsize': (10, 8)})
# corr = oversampled_red_data.corr()
# sns.heatmap(corr,
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values,
#             cmap="YlGnBu",
#             annot=True,
#             fmt=".2f")
# plt.show()

# %% [markdown]
# ## File name:

# %% [code]
# FN = File Name


normalized_clean_data_FN = 'normalized_clean_data_red.csv'
oversampled_clean_data_FN = 'oversampled_clean_data_red.csv'
clean_data_FN = 'clean_red.csv'

# normalized_clean_data_FN = 'normalized_clean_data_white.csv'
# oversampled_clean_data_FN = 'oversampled_clean_data_white.csv'
# clean_data_FN = 'clean_white.csv'

# %% [code]
# %matplotlib inline

# %% [code]
data = pd.read_csv(normalized_clean_data_FN)
X = data.drop(['quality'], axis=1)
Y = data['quality']
data.head()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

Y.hist()
plt.show()

# %% [markdown]
# # Question 1.a - Regrestion
# ##### we'll try out three regresion models and compair them.
# ##### 1. linear and polynomial regrestion
# ##### 2. SVM Regresor
# ##### 3. neural network regresor

# %% [markdown]
# ## Linear Regrestion

# %% [code]
reg = LinearRegression().fit(X_train, Y_train)
print('linear regression r2 score = ', reg.score(X_test, Y_test))

a = reg.predict(X_train)
train_mse = (mean_squared_error(a, Y_train))
print('linear regression train_mse =', train_mse)
b = reg.predict(X_test)
test_mse = (mean_squared_error(b, Y_test))
print('linear regression test_mse =', test_mse)

# %% [markdown]
# ## Polynomial Regression

# %% [code]
p_reg = Pipeline([('poly', PolynomialFeatures(degree=3)),
                  ('linear', LinearRegression(fit_intercept=False))])

p_reg = p_reg.fit(X_train, Y_train)
p_reg.named_steps['linear'].coef_

from sklearn.metrics import mean_squared_error

print('polynomial regression r2 score = ', p_reg.score(X_test, Y_test))
a = p_reg.predict(X_train)
train_mse = (mean_squared_error(a, Y_train))
print('polynomial regression train_mse =', train_mse)
b = p_reg.predict(X_test)
test_mse = (mean_squared_error(b, Y_test))
print('polynomial regression test_mse =', test_mse)

# %% [markdown]
# ## SVM Regresors
# #### 1. rbf kernel
# #### 2. linear kernel
# #### 3. polynomial kernel

# %% [code]
SVR_rbf = svm.SVR(kernel='rbf')
SVR_lin = svm.SVR(kernel='linear')
SVR_poly = svm.SVR(kernel='poly', degree=2)
SVR_rbf.fit(X_train, Y_train)
SVR_lin.fit(X_train, Y_train)
SVR_poly.fit(X_train, Y_train)

print('SVR_rbf score = ', SVR_rbf.score(X_train, Y_train))
a = SVR_rbf.predict(X_train)
train_mse = (mean_squared_error(a, Y_train))
print('SVR_rbf train_mse = ', train_mse)
b = SVR_rbf.predict(X_test)
test_mse = (mean_squared_error(b, Y_test))
print('SVR_rbf test_mse = ', test_mse, '\n')

print('SVR_lin score = ', SVR_lin.score(X_train, Y_train))
a = SVR_lin.predict(X_train)
train_mse = (mean_squared_error(a, Y_train))
print('SVR_lin train_mse = ', train_mse)
b = SVR_lin.predict(X_test)
test_mse = (mean_squared_error(b, Y_test))
print('SVR_lin test_mse = ', test_mse, '\n')

print('SVR_poly score = ', SVR_poly.score(X_train, Y_train))
a = SVR_poly.predict(X_train)
train_mse = (mean_squared_error(a, Y_train))
print('SVR_poly train_mse = ', train_mse)
b = SVR_poly.predict(X_test)
test_mse = (mean_squared_error(b, Y_test))
print('SVR_poly test_mse = ', test_mse)

# %% [markdown]
# # Deep Neural Network Regresor

# %% [code]
import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.layers import Dropout
from keras.models import Model
import tensorflow as tf

# %% [code]
DNN_model_selu = Sequential()
DNN_model_selu.add(Dense(100, input_dim=11, activation="selu"))
DNN_model_selu.add(Dropout(0.1))
DNN_model_selu.add(Dense(128, activation="selu"))
DNN_model_selu.add(Dropout(0.1))
DNN_model_selu.add(Dense(128, activation="selu"))
DNN_model_selu.add(Dropout(0.1))
DNN_model_selu.add(Dense(2, activation="selu"))
DNN_model_selu.add(Dense(1))

# %% [code]
DNN_model_relu = Sequential()
DNN_model_relu.add(Dense(100, input_dim=11, activation="relu"))
DNN_model_relu.add(Dropout(0.1))
DNN_model_relu.add(Dense(128, activation="relu"))
DNN_model_relu.add(Dropout(0.1))
DNN_model_relu.add(Dense(128, activation="relu"))
DNN_model_relu.add(Dropout(0.1))
DNN_model_relu.add(Dense(2, activation="relu"))
DNN_model_relu.add(Dense(1))

# %% [code]
DNN_model_soft_max = Sequential()
DNN_model_soft_max.add(Dense(100, input_dim=11, activation="softmax"))
DNN_model_soft_max.add(Dropout(0.1))
DNN_model_soft_max.add(Dense(128, activation="softmax"))
DNN_model_soft_max.add(Dropout(0.1))
DNN_model_soft_max.add(Dense(128, activation="softmax"))
DNN_model_soft_max.add(Dropout(0.1))
DNN_model_soft_max.add(Dense(2, activation="softmax"))
DNN_model_soft_max.add(Dense(1))

# %% [code]
DNN_model_selu.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
# history = NN_model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
selu_history = DNN_model_selu.fit(X_train, Y_train, epochs=100, validation_split=0.3)

# %% [code]
DNN_model_relu.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
# history = NN_model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
relu_history = DNN_model_relu.fit(X_train, Y_train, epochs=100, validation_split=0.3)

# %% [code]
DNN_model_soft_max.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
# history = NN_model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
soft_max_history = DNN_model_soft_max.fit(X_train, Y_train, epochs=100, validation_split=0.3)

# %% [code]
# list all data in history
print(selu_history.history.keys())
print(relu_history.history.keys())
# summarize history for accuracy
plt.figure(figsize=(10, 5))
plt.plot(selu_history.history['mean_squared_error'], label='selu')
# plt.plot(selu_history.history['val_mean_squared_error'], label='selu')
plt.plot(relu_history.history['mean_squared_error'], label='relu')
# plt.plot(relu_history.history['val_mean_squared_error'], label='relu')
plt.plot(soft_max_history.history['mean_squared_error'], label='soft max')
# plt.plot(soft_max_history.history['val_mean_squared_error'], label='soft max')
plt.title('mean_squared_error')
plt.ylabel('mean_squared_error')
plt.xlabel('epoch')
plt.legend()
plt.show()
# # summarize history for loss
# plt.figure(figsize = (12,7))
# plt.plot(selu_history.history['loss'], label='selu')
# # plt.plot(selu_history.history['val_loss'], label='selu')
# plt.plot(relu_history.history['loss'], label='relu')
# # plt.plot(relu_history.history['val_loss'], label='relu')
# plt.plot(soft_max_history.history['loss'], label='soft max')
# # plt.plot(soft_max_history.history['val_loss'], label='soft max')
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend()
# plt.show()

# %% [code]
a_selu = DNN_model_selu.predict(X_train)
train_mse = (mean_squared_error(a_selu, Y_train))
print('selu NN train_mse = ', train_mse)
b_selu = DNN_model_selu.predict(X_test)
test_mse = (mean_squared_error(b_selu, Y_test))
print('selu NN test_mse = ', test_mse)

a_relu = DNN_model_relu.predict(X_train)
train_mse = (mean_squared_error(a_relu, Y_train))
print('relu NN train_mse = ', train_mse)
b_relu = DNN_model_relu.predict(X_test)
test_mse = (mean_squared_error(b_relu, Y_test))
print('relu NN test_mse = ', test_mse)

a_soft_max = DNN_model_soft_max.predict(X_train)
train_mse = (mean_squared_error(a_relu, Y_train))
print('soft max NN train_mse = ', train_mse)
b_soft_max = DNN_model_soft_max.predict(X_test)
test_mse = (mean_squared_error(b_relu, Y_test))
print('soft max NN test_mse = ', test_mse)


# %% [code]

# # # # # # # # # # # # # # # # # # # # # # # Clustering:
def points_for_cluster_df(clustering_labels):
    hist_dictionary = {}
    for i in clustering_labels:
        if str(i) in hist_dictionary:
            hist_dictionary[str(i)] = hist_dictionary[str(i)] + 1
        else:
            hist_dictionary[str(i)] = 1
    clusters_names_numeric = [int(x) for x in hist_dictionary.keys()]
    clusters_names_numeric.sort()
    clusters_names = [str(name) for name in clusters_names_numeric]
    points_amount_for_cluster = [hist_dictionary[x] for x in clusters_names]
    clustering_hist_df = pd.DataFrame([points_amount_for_cluster])
    clustering_hist_df.columns = clusters_names
    return clustering_hist_df


def plot_silhouette_score(clustering_method, clusters, s_scores):
    plt.plot(clusters, s_scores, color='blue', label='Silhouette score')
    plt.title(clustering_method + ' clustering Silhouette score depending on clusters number')
    plt.ylabel('Silhouette score')
    plt.xlabel('clusters')
    # plt.axhline(0, lw=0.5, color='black')
    # plt.axvline(0, lw=0.5, color='black')
    plt.legend()
    plt.show()


# %% [markdown]
# # BIRCH

# %% [code]
birch_s_scores = []
clusters = range(2, 180)
for i in clusters:
    br = Birch(n_clusters=i).fit(X_train)
    birch_s_scores.append(silhouette_score(X_train, br.labels_))

# %% [code]
plot_silhouette_score('Birch', clusters, birch_s_scores)

# %% [code]
# Optimize Birch clustering with 105 clusters:
optimize_br = Birch(n_clusters=105).fit(X_train)
points_for_cluster_df(optimize_br.labels_)

# %% [markdown]
# # K-Means

# %% [code]
km_s_scores = []
clusters = range(2, 180)
for i in clusters:
    k_means = KMeans(n_clusters=i, random_state=0).fit(X_train)
    km_s_scores.append(silhouette_score(X_train, k_means.labels_))

# %% [code]
plot_silhouette_score('K-Means', clusters, km_s_scores)

# %% [code]
# Optimize K-Means clustering 105 clusters:

optimize_km = KMeans(n_clusters=105, random_state=0).fit(X_train)
points_for_cluster_df(optimize_km.labels_)

# %% [markdown]
# # MeanShift

# %% [code]
# Optimize MeanShift clustering using estimate_bandwidth:

optimize_bandwidth = estimate_bandwidth(X_train)
optimize_ms = MeanShift(bandwidth=optimize_bandwidth).fit(X_train)
points_for_cluster_df(optimize_ms.labels_)

print(optimize_bandwidth)
print()


# %% [markdown]
# ### Predict qualities with the clusters mean quality

# %% [code]
def mean_quality_for_cluster_dictionary(df, clustering):
    mean_qualities = {}
    for i in df[clustering].unique():
        mean_qualities[i] = df[df[clustering] == i]['quality'].mean()
    return mean_qualities


def cluster_mse_score(cluster_quality_dict: dict, test_cluster_lables: pd.Series, real_test_values: pd.Series):
    predicted_qualities = []
    for i in test_cluster_lables:
        predicted_qualities.append(cluster_quality_dict[i])
    predicted_qualities = pd.Series(predicted_qualities)
    return mean_squared_error(predicted_qualities, real_test_values)


train_with_clusters = X_train.copy()
train_with_clusters['quality'] = Y_train
# MeanShift:
train_with_clusters['ms_cluster'] = optimize_ms.labels_
ms_qualities_dict = mean_quality_for_cluster_dictionary(train_with_clusters, 'ms_cluster')
# Birch:
train_with_clusters['br_cluster'] = optimize_br.labels_
br_qualities_dict = mean_quality_for_cluster_dictionary(train_with_clusters, 'br_cluster')
# Kmeans
train_with_clusters['km_cluster'] = optimize_km.labels_
km_qualities_dict = mean_quality_for_cluster_dictionary(train_with_clusters, 'km_cluster')

print()
print('br_qualities_dict: ', br_qualities_dict)
print()
print('km_qualities_dict: ', km_qualities_dict)
print()
print('ms_qualities_dict: ', ms_qualities_dict)
print()

test = X_test.copy()
test['quality'] = Y_test

# Birch:
br_test_labels = optimize_br.predict(X_test)
print('Birch mse: ', cluster_mse_score(br_qualities_dict, br_test_labels, test['quality']))

# K-Means
km_test_labels = optimize_km.predict(X_test)
print('K-Means mse: ', cluster_mse_score(km_qualities_dict, km_test_labels, test['quality']))

# MeanShift:
ms_test_labels = optimize_ms.predict(X_test)
print('Mean Shift mse: ', cluster_mse_score(ms_qualities_dict, ms_test_labels, test['quality']))


# %% [markdown]
# # Dimension Reduction

# %% [code]
def printPrejections(reducted_df: pd.DataFrame, model_name: str):
    for i in range(0, 2):
        for j in range(i + 1, 3):
            plotName = model_name + ' (dim ' + str(i) + '),(dim ' + str(j) + ')'
            dim1 = '(dim ' + str(i) + ')'
            dim2 = '(dim ' + str(j) + ')'
            vis = pd.DataFrame()
            vis[dim1] = reducted_df[i]
            vis[dim2] = reducted_df[j]
            vis['Y'] = reducted_df['quality']
            plt.figure(figsize=(16, 10))
            sns.scatterplot(
                x=dim1, y=dim2,
                hue="Y",
                palette=sns.color_palette("hls", len(set(list(reducted_df['quality'])))),
                data=vis,
                legend="full",
                alpha=0.8
            ).set_title(plotName)
            plt.show()


# %% [markdown]
# ## pca

# %% [code]
from sklearn.decomposition import PCA, IncrementalPCA

pca = IncrementalPCA(n_components=3)
pca.fit(X)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)

pca_df = pd.DataFrame(pca.transform(X))
pca_df['quality'] = Y

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

# %% [markdown]
# ## Auto Encoder

# %% [code]
input_dim = Input(shape=(11,))
# DEFINE THE DIMENSION OF ENCODER ASSUMED 3
encoding_dim = 3
# DEFINE THE ENCODER LAYERS
encoded1 = Dense(9, activation='relu')(input_dim)
encoded2 = Dense(6, activation='relu')(encoded1)
encoded3 = Dense(4, activation='relu')(encoded2)
encoded4 = Dense(encoding_dim, activation='relu')(encoded3)
# DEFINE THE DECODER LAYERS
decoded1 = Dense(4, activation='relu')(encoded4)
decoded2 = Dense(6, activation='relu')(decoded1)
decoded3 = Dense(9, activation='relu')(decoded2)
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
AE_data['quality'] = Y_test.values

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

# %% [markdown]
# ## LLE

# %% [code]
from sklearn.manifold import LocallyLinearEmbedding

embedding = LocallyLinearEmbedding(n_components=3)
X_transformed = embedding.fit_transform((X - X.mean()) / X.std())
X_transformed.shape

LLE_data = pd.DataFrame(X_transformed)
LLE_data['quality'] = Y.values
import seaborn as sns

sns.set(rc={'figure.figsize': (10, 8)})
corr = LLE_data.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap="YlGnBu",
            annot=True,
            fmt=".2f")
plt.show()

printPrejections(LLE_data, 'LLE')

# %% [markdown]
# # Linear Models Compereson

# %% [code]
import math


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def continues_jacard(vec1: np.ndarray, vec2: np.ndarray):
    numerator = 0
    denominator = 0
    if len(vec1) == len(vec2):
        for i in range(0, len(vec1)):
            numerator += min(vec1[i], vec2[i])
            denominator += max(vec1[i], vec2[i])
        return numerator / denominator
    return -1


def vectorComperesion(vec1, vec1_name: str, vec2, vec2_name: str):
    print(f'radial angle between {vec1_name} and {vec2_name} = {angle(vec1, vec2)}')
    print(f'***cosine test between {vec1_name} and {vec2_name} = {math.cos(angle(vec1, vec2))}***')
    print(f'euclidean distance between {vec1_name} and {vec2_name} = {np.linalg.norm(vec1 - vec2)}')
    jac = continues_jacard(vec1, vec2)
    if jac != -1:
        print(f'jaccard score between {vec1_name} and {vec2_name} = {jac}')
    else:
        print(f'jaccard Error for {vec1_name} and {vec2_name}')


def threeVectorComperesion(vec1, vec1_name: str, vec2, vec2_name: str, vec3, vec3_name: str):
    print(f'{vec1_name} is {vec1}')
    print(f'{vec2_name} is {vec2}')
    print(f'{vec3_name} is {vec3}\n\n')

    print(f'magnitude of {vec1_name} is {np.linalg.norm(vec1)}')
    print(f'magnitude of {vec2_name} is {np.linalg.norm(vec2)}')
    print(f'magnitude of {vec3_name} is {np.linalg.norm(vec3)}\n\n')
    vec_list = [(vec1, vec1_name), (vec2, vec2_name), (vec3, vec3_name)]
    for i in range(0, 2):
        for j in range(i + 1, 3):
            vectorComperesion(vec_list[i][0], vec_list[i][1], vec_list[j][0], vec_list[j][1])


# %% [code]
red_file = 'clean_red.csv'
white_file = 'clean_white.csv'
red_data = pd.read_csv(red_file)
white_data = pd.read_csv(white_file)
all_data = pd.concat([red_data, white_data])

# %% [code]
all_data.head()

# %% [code]
all_mean = all_data.drop(['quality'], axis=1).mean()
all_std = all_data.drop(['quality'], axis=1).std()
normalized_all_X = (all_data.drop(['quality'], axis=1) - all_mean) / all_std

normalized_red_X = (red_data.drop(['quality'], axis=1) - all_mean) / all_std
normalized_white_X = (white_data.drop(['quality'], axis=1) - all_mean) / all_std

red_X_train, red_X_test, red_y_train, red_y_test = \
    train_test_split(normalized_red_X, red_data['quality'], test_size=0.20, random_state=42)
white_X_train, white_X_test, white_y_train, white_y_test = \
    train_test_split(normalized_white_X, white_data['quality'], test_size=0.20, random_state=42)
all_X_train, all_X_test, all_y_train, all_y_test = \
    train_test_split(normalized_all_X, all_data['quality'], test_size=0.20, random_state=42)

# %% [code]
# Linear Regrestion

red_lin_reg = LinearRegression().fit(red_X_train, red_y_train)
white_lin_reg = LinearRegression().fit(white_X_train, white_y_train)
all_lin_reg = LinearRegression().fit(all_X_train, all_y_train)

red_lin_reg_vec = np.array(red_lin_reg.coef_)
white_lin_reg_vec = np.array(white_lin_reg.coef_)
all_lin_reg_vec = np.array(all_lin_reg.coef_)

threeVectorComperesion(red_lin_reg_vec, 'red_lin_reg_vec', white_lin_reg_vec, 'white_lin_reg_vec',
                       all_lin_reg_vec, 'all_lin_reg_vec')

# %% [code]
# linear SVM


red_lin_SVM = svm.SVR(kernel='linear').fit(red_X_train, red_y_train)
white_lin_SVM = svm.SVR(kernel='linear').fit(white_X_train, white_y_train)
all_lin_SVM = svm.SVR(kernel='linear').fit(all_X_train, all_y_train)

red_lin_SVM_vec = np.array(red_lin_SVM.coef_[0])
white_lin_SVM_vec = np.array(white_lin_SVM.coef_[0])
all_lin_SVM_vec = np.array(all_lin_SVM.coef_[0])

threeVectorComperesion(red_lin_SVM_vec, 'red_lin_SVM_vec', white_lin_SVM_vec, 'white_lin_SVM_vec',
                       all_lin_SVM_vec, 'all_lin_SVM_vec')

# %% [code]
# PCA
red_pca = IncrementalPCA(n_components=3)
white_pca = IncrementalPCA(n_components=3)
all_pca = IncrementalPCA(n_components=3)

red_pca.fit(normalized_red_X)
white_pca.fit(normalized_white_X)
all_pca.fit(normalized_all_X)

sns.set(rc={'figure.figsize': (10, 8)})
corr = pd.DataFrame(np.corrcoef(red_pca.components_, white_pca.components_),
                    columns=['red0', 'red1', 'red2', 'white0', 'white1', 'white2'])
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap="YlGnBu",
            annot=True,
            fmt=".2f")
plt.show()
red_pca_vec_arr = [np.array(red_pca.components_[0]), np.array(red_pca.components_[1]), np.array(red_pca.components_[2])]
white_pca_vec_arr = [np.array(white_pca.components_[0]), np.array(white_pca.components_[1]),
                     np.array(white_pca.components_[2])]
all_pca_vec_arr = [np.array(all_pca.components_[0]), np.array(all_pca.components_[1]), np.array(all_pca.components_[2])]


# %% [code]

def compareTwoPCAs(PCA_vecs1, PCA_vecs1_name: str, PCA_vecs2, PCA_vecs2_name: str):
    table = []
    for i in range(0, len(PCA_vecs1)):
        table.append([])
        for j in range(0, len(PCA_vecs2)):
            table[i].append(math.cos(angle(PCA_vecs1[i], PCA_vecs2[j])))
    sns.set(rc={'figure.figsize': (10, 8)})
    df = pd.DataFrame(table, columns=[PCA_vecs2_name + ' 0', PCA_vecs2_name + ' 1', PCA_vecs2_name + ' 2'],
                      index=[PCA_vecs1_name + ' 0', PCA_vecs1_name + ' 1', PCA_vecs1_name + ' 2'])
    sns.heatmap(df,
                cmap="YlGnBu",
                annot=True,
                fmt=".2f")
    plt.show()


compareTwoPCAs(red_pca_vec_arr, 'red Pca ', white_pca_vec_arr, 'white Pca ')
plt.show()
compareTwoPCAs(red_pca_vec_arr, 'red Pca ', all_pca_vec_arr, 'all Pca ')
plt.show()
compareTwoPCAs(white_pca_vec_arr, 'white PCA ', all_pca_vec_arr, 'all Pca ')
plt.show()


# %% [markdown]
# # Question number 3

# %% [markdown]
# ## 3A- Exploring the learning ease of each attribute:

# %% [code]
class Combination:
    def __init__(self, att_comb, mse):
        self.att_comb = att_comb
        self.mse = mse

    def get_att_comb(self):
        return self.att_comb

    def get_mse(self):
        return self.mse

    def get_number_of_atributes(self):
        return len(self.att_comb)


def calc_lin_reg_mse(X, combination, attribute_df) -> float:
    X_train, X_test, attribute_train, attribute_test = train_test_split(X[[col for col in combination]],
                                                                        attribute_df, test_size=0.18, random_state=42)
    reg = LinearRegression().fit(X_train, attribute_train)
    attribute_pred = reg.predict(X_test)

    return mean_squared_error(attribute_test, attribute_pred)


def open_tuple_with_dict(dictionary, tup):
    vals_list = [dictionary[i] for i in tup]
    return tuple(vals_list)


# %% [code]


# # Linear regresion MSE of attribute,
# # depending on the number of attributes used in the learning process

attribute_combinations_list = []
for attribute in X.columns.tolist():
    attribute_df = X[attribute]
    current_X = X.drop([attribute], axis=1)
    attributes = X.columns.tolist()
    attributes.remove(attribute)
    attributes_dict = {}
    for i in range(10):
        attributes_dict[i] = attributes[i]
    current_X.columns = range(10)
    combinations_list = []
    for i in range(10):
        min_mse = float('inf')
        min_combination = ()
        for combination in combinations(range(10), i + 1):
            lin_reg_calc = calc_lin_reg_mse(current_X, combination, attribute_df)
            if lin_reg_calc < min_mse:
                min_mse = lin_reg_calc
                min_combination = combination
        min_combination = open_tuple_with_dict(attributes_dict, min_combination)
        combinations_list.append(Combination(min_combination, min_mse))
    attribute_combinations_list.append((attribute, combinations_list))

attribute = 'quality'
attribute_df = Y.copy()
current_X = X.copy()
attributes = X.columns.tolist()
attributes_dict = {}
for i in range(11):
    attributes_dict[i] = attributes[i]
current_X.columns = range(11)
combinations_list = []
for i in range(11):
    min_mse = float('inf')
    min_combination = ()
    for combination in combinations(range(11), i + 1):
        lin_reg_calc = calc_lin_reg_mse(current_X, combination, attribute_df)
        if lin_reg_calc < min_mse:
            min_mse = lin_reg_calc
            min_combination = combination
    min_combination = open_tuple_with_dict(attributes_dict, min_combination)
    combinations_list.append(Combination(min_combination, min_mse))
attribute_combinations_list.append((attribute, combinations_list))

# %% [code]

fig, axs = plt.subplots(6, 2, figsize=(25, 45))
for attribute in range(len(X.columns.tolist())):
    xs = [i + 1 for i in range(10)]
    ys = [comb.get_mse() for comb in attribute_combinations_list[attribute][1]]
    axs[int(attribute / 2), int(attribute % 2)].set_xlabel('number of attributes', fontsize=16)
    axs[int(attribute / 2), int(attribute % 2)].set_ylabel('min mse', fontsize=16)
    #     axs[int(attribute / 2), int(attribute % 2)].axhline(0, lw=0.5, color='black')
    #     axs[int(attribute / 2), int(attribute % 2)].axvline(0, lw=0.5, color='black')
    axs[int(attribute / 2), int(attribute % 2)].set_title(attribute_combinations_list[attribute][0], fontsize=24)
    axs[int(attribute / 2), int(attribute % 2)].plot(xs, ys)
# quality:
attribute = 11
xs = [i + 1 for i in range(11)]
ys = [comb.get_mse() for comb in attribute_combinations_list[attribute][1]]
axs[int(attribute / 2), int(attribute % 2)].set_xlabel('number of attributes', fontsize=16)
axs[int(attribute / 2), int(attribute % 2)].set_ylabel('min mse', fontsize=16)
#     axs[int(attribute / 2), int(attribute % 2)].axhline(0, lw=0.5, color='black')
#     axs[int(attribute / 2), int(attribute % 2)].axvline(0, lw=0.5, color='black')
axs[int(attribute / 2), int(attribute % 2)].set_title(attribute_combinations_list[attribute][0], fontsize=24)
axs[int(attribute / 2), int(attribute % 2)].plot(xs, ys)
##
fig.tight_layout(pad=10.0)

# %% [code]

print('for ' + data.columns.tolist()[i] + ':')
for comb in attribute_combinations_list[i][1]:
    print('    for ' + str(comb.get_number_of_atributes()) + ' attributes combination: ')
    print('    the min mse is: ' + str(comb.get_mse()) + ', with the combination: ')
    print('    ' + str(comb.get_att_comb()))

# %% [code]


# # 3B- Optimal result for the attributes learning
# Relies on the graphs above, using the elbow method.

def learn_attribute(attribute_placement, num_of_attributes, attribute_combinations_list):
    print('For ' + attribute_combinations_list[attribute_placement][0] + ',')
    print('we got the optimal result when learning with ' + str(num_of_attributes) + ' attributes,')
    print('and those attributes are: ' +
          str(attribute_combinations_list[attribute_placement][1][num_of_attributes - 1].get_att_comb()))


def print_optimal_learning_results(num_of_att_list, attribute_combinations_list):
    for i in range(11):
        learn_attribute(i, num_of_att_list[i], attribute_combinations_list)
        print()


num_of_att_list = [4, 3, 2, 4, 6, 9, 7, 4, 4, 5, 5]

print_optimal_learning_results(num_of_att_list, attribute_combinations_list)

n = 11

colors = {}
colors_list = ['#FF0000', '#FF00A2', '#C332EF', '#2B00FF', '#00EFFF', '#29CD44', '#EFFF00', '#E9A70D',
               '#6E7F00', '#7F007A', '#000000']

attribute_to_index = {}

for i in range(len(X.columns.tolist())):
    colors[i] = colors_list[i]
    attribute_to_index[X.columns.tolist()[i]] = i

G = nx.MultiDiGraph()
columns = X.columns.tolist()
G.add_nodes_from(columns)

node_list = G.nodes()
angle = []
angle_dict = {}
for i, node in zip(range(n), node_list):
    theta = 2.0 * np.pi * i / n
    angle.append((np.cos(theta), np.sin(theta)))
    angle_dict[node] = theta

for attribute_placement in range(len(columns)):
    for j in range(num_of_att_list[attribute_placement]):
        G.add_edges_from([(attribute_combinations_list[attribute_placement][0],
                           attribute_combinations_list[attribute_placement][1][
                               num_of_att_list[attribute_placement] - 1].get_att_comb()[j])])

existence_matrix = [[0 for j in range(11)] for i in range(11)]
for e in G.edges:
    num_of_same_edges = max(existence_matrix[attribute_to_index[e[0]]][attribute_to_index[e[1]]],
                            existence_matrix[attribute_to_index[e[1]]][attribute_to_index[e[0]]])
    existence_matrix[attribute_to_index[e[0]]][attribute_to_index[e[1]]] = num_of_same_edges + 1
    existence_matrix[attribute_to_index[e[1]]][attribute_to_index[e[0]]] = num_of_same_edges + 1


def edge_num(att1, att2):
    edge_num = existence_matrix[attribute_to_index[att1]][attribute_to_index[att2]] - 1
    existence_matrix[attribute_to_index[att1]][attribute_to_index[att2]] -= 1
    existence_matrix[attribute_to_index[att2]][attribute_to_index[att2]] -= 1
    return edge_num


color_map = []
for node in G:
    color_map.append(colors[attribute_to_index[node]])

# figsize is intentionally set small to condense the graph
fig, ax = plt.subplots(figsize=(5, 5))
margin = 0.1
fig.subplots_adjust(margin, margin, 1. - margin, 1. - margin)
ax.axis('equal')

pos = nx.shell_layout(G)
nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=100, alpha=1, with_labels=True)
ax = plt.gca()

for e in G.edges:
    ax.annotate("",
                xy=pos[e[0]], xycoords='data',
                xytext=pos[e[1]], textcoords='data',
                arrowprops=dict(arrowstyle="<-", color=colors[attribute_to_index[e[0]]],
                                shrinkA=15, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="arc3,rad=rrr".replace('rrr', str(0.3 * edge_num(e[0], e[1]))
                                                                       ),
                                ),
                )

description = nx.draw_networkx_labels(G, pos)

factor_dic = {
    'fixed_acidity': (0.09, 0),
    'volatile_acidity': (0.1, -0.02),
    'citric_acid': (0, -0.03),
    'residual_sugar': (0, -0.04),
    'chlorides': (0, -0.01),
    'free_sulfur_dioxide': (0, 0.05),
    'total_sulfur_dioxide': (0, -0.05),
    'density': (0, 0),
    'pH': (0, 0.03),
    'sulphates': (0, 0.025),
    'alcohol': (0, 0),
}

r = fig.canvas.get_renderer()
trans = plt.gca().transData.inverted()
for node, t in description.items():
    bb = t.get_window_extent(renderer=r)
    bbdata = bb.transformed(trans)
    radius = 1 + 0.1
    x_factor = factor_dic[str(node)][0]
    y_factor = factor_dic[str(node)][1]
    position = (radius * np.cos(angle_dict[node]) + x_factor, radius * np.sin(angle_dict[node]) + y_factor)
    t.set_position(position)
    t.set_clip_on(False)

plt.axis('off')
plt.show()

# %% [code]
learning_features = {}
for i in range(11):
    comb = attribute_combinations_list[i][1][num_of_att_list[i]]
    learning_features[data.columns.tolist()[i]] = comb.get_att_comb()
    print('for ' + data.columns.tolist()[i] + ':')
    print('    for ' + str(comb.get_number_of_atributes()) + ' attributes combination: ')
    print('    the min mse is: ' + str(comb.get_mse()) + ', with the combination: ')
    print('    ' + str(comb.get_att_comb()))

print()
print(learning_features)
learning_features_list_of_tuples = [learning_features[feature] for feature in learning_features]
learning_features_list_of_lists = [[feature for feature in tup] for tup in learning_features_list_of_tuples]
learning_features_df = pd.DataFrame(learning_features_list_of_lists).T
learning_features_df.columns = [feature for feature in learning_features]
learning_features_df = learning_features_df.fillna(value='')


# %% [code]
def learning_score(number_of_features, MSE):
    return math.sqrt(number_of_features) * MSE


num_of_att_list = [4, 3, 2, 4, 6, 9, 7, 4, 4, 5, 5]

learning_dict = {}
for i in range(0, 11):
    comb = attribute_combinations_list[i][1][num_of_att_list[i]]
    learning_dict[data.columns.tolist()[i]] = learning_score(comb.get_number_of_atributes(), comb.get_mse())
    print(
        f'the learning score for {data.columns.tolist()[i]} is:  {learning_score(comb.get_number_of_atributes(), comb.get_mse())}')

print({k: v for k, v in sorted(learning_dict.items(), key=lambda item: item[1])})

# %% [markdown]
# # World's greatest grape

# %% [code]
unnormal = pd.read_csv(oversampled_clean_data_FN).drop(['quality'], axis=1)
mean = unnormal.mean()
std = unnormal.std()


# %% [code]
def CustumLossFunctionSVM_rbf(x: list):
    X_df = pd.DataFrame([x], columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                                      'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
                                      'density', 'pH', 'sulphates', 'alcohol'])
    prediction_value = SVR_rbf.predict(X_df)
    while type(prediction_value) == np.ndarray:
        prediction_value = prediction_value[0]
    return abs(10 - prediction_value)


def CustumLossFunctionDNN(x: list):
    X_df = pd.DataFrame([x], columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                                      'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
                                      'density', 'pH', 'sulphates', 'alcohol'])
    prediction_value = DNN_model_relu.predict(X_df)
    while type(prediction_value) == np.ndarray:
        prediction_value = prediction_value[0]
    return abs(10 - prediction_value)


methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr']

# %% [code]
min_error = 100
best_method = ''
x0 = list(data[data.quality == data.quality.max()].mean().to_frame().T.drop(['quality'], axis=1).values[0])
print(x0)
for m in methods:
    res = minimize(CustumLossFunctionSVM_rbf, x0, method=m, tol=1e-6, options={'maxiter': 10000})
    if res.fun < min_error:
        best_method = m
        min_error = res.fun

print(best_method)
print(min_error)
res = minimize(CustumLossFunctionSVM_rbf, x0, method=best_method, tol=1e-6, options={'maxiter': 10000})
print('best grape SVR_rbf score: ', SVR_rbf.predict(pd.Series(res.x).to_frame().T)[0])
print('best grape DNN relu score: ', DNN_model_relu.predict(pd.Series(res.x).to_frame().T)[0][0])
print('best grape SVR_lin score: ', SVR_lin.predict(pd.Series(res.x).to_frame().T)[0])
print('best grape SVR_poly score: ', SVR_poly.predict(pd.Series(res.x).to_frame().T)[0])
print('best grape Linear regression score: ', reg.predict(pd.Series(res.x).to_frame().T)[0])

# %% [code]
pd.DataFrame([res.x], columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                               'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
                               'density', 'pH', 'sulphates', 'alcohol']) * std + mean

# %% [code]
min_error = 100
best_method = ''
x0 = list(data[data.quality == data.quality.max()].mean().to_frame().T.drop(['quality'], axis=1).values[0])
print(x0)
for m in methods:
    res = minimize(CustumLossFunctionDNN, x0, method=m, tol=1e-6, options={'maxiter': 10000})
    if res.fun < min_error:
        best_method = m
        min_error = res.fun

print(best_method)
print(min_error)
res = minimize(CustumLossFunctionDNN, x0, method=best_method, tol=1e-6, options={'maxiter': 10000})
print('best grape SVR_rbf score: ', SVR_rbf.predict(pd.Series(res.x).to_frame().T)[0])
print('best grape DNN relu score: ', DNN_model_relu.predict(pd.Series(res.x).to_frame().T)[0][0])
print('best grape SVR_lin score: ', SVR_lin.predict(pd.Series(res.x).to_frame().T)[0])
print('best grape SVR_poly score: ', SVR_poly.predict(pd.Series(res.x).to_frame().T)[0])
print('best grape Linear regression score: ', reg.predict(pd.Series(res.x).to_frame().T)[0])

# %% [code]
pd.DataFrame([res.x], columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                               'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
                               'density', 'pH', 'sulphates', 'alcohol']) * std + mean

# %% [code]
min_error = 100
best_method = ''
x0 = list(data[data.quality == data.quality.max()].mean().to_frame().T.drop(['quality'], axis=1).values[0])
print(x0)
for m in methods:
    res = minimize(CustumLossFunctionSVM_rbf, x0, method=m, tol=1e-6, options={'maxiter': 10000})
    if res.fun < min_error:
        best_method = m
        min_error = res.fun

print(best_method)
print(min_error)
res = minimize(CustumLossFunctionSVM_rbf, x0, method=best_method, tol=1e-6, options={'maxiter': 10000})
print('best grape SVR_rbf score: ', SVR_rbf.predict(pd.Series(res.x).to_frame().T)[0])
print('best grape DNN relu score: ', DNN_model_relu.predict(pd.Series(res.x).to_frame().T)[0][0])
print('best grape SVR_lin score: ', SVR_lin.predict(pd.Series(res.x).to_frame().T)[0])
print('best grape SVR_poly score: ', SVR_poly.predict(pd.Series(res.x).to_frame().T)[0])
print('best grape Linear regression score: ', reg.predict(pd.Series(res.x).to_frame().T)[0])

# %% [code]
pd.DataFrame([res.x], columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                               'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
                               'density', 'pH', 'sulphates', 'alcohol']) * std + mean

# %% [code]
# learning_methods = ['Linear_Regression', 'SVM_rbf', 'SVM_linear', 'SVM_poly']

# %% [code]
# def calc_learning_mse(X, combination, attribute_df):
#     X_train, X_test, attribute_train, attribute_test = train_test_split(X[[col for col in combination]],
#                                                                         attribute_df, test_size=0.18, random_state=42)

#     reg = LinearRegression().fit(X_train, attribute_train)
#     attribute_pred = reg.predict(X_test)
#     lin_reg_mse = mean_squared_error(attribute_test, attribute_pred)

#     reg = svm.SVR(kernel='rbf').fit(X_train, attribute_train)
#     attribute_pred = reg.predict(X_test)
#     svm_rbf_mse = mean_squared_error(attribute_test, attribute_pred)

#     reg = svm.SVR(kernel='linear').fit(X_train, attribute_train)
#     attribute_pred = reg.predict(X_test)
#     svm_linear_mse = mean_squared_error(attribute_test, attribute_pred)

#     reg = svm.SVR(kernel='poly', degree = 2).fit(X_train, attribute_train)
#     attribute_pred = reg.predict(X_test)
#     svm_poly_mse = mean_squared_error(attribute_test, attribute_pred)

#     mse_dict = {
#         learning_methods[0]: lin_reg_mse,
#         learning_methods[1]: svm_rbf_mse,
#         learning_methods[2]: svm_linear_mse,
#         learning_methods[3]: svm_poly_mse,
#     }

#     return mse_dict

# %% [code]
# attribute = 'quality'
# attribute_df = Y.copy()
# current_X = X.copy()
# attributes = X.columns.tolist()
# attributes_dict = {}
# for i in range(11):
#     attributes_dict[i] = attributes[i]
# current_X.columns = range(11)
# learning_method_combination_list = []
# for i in range(len(learning_methods)):
#     learning_method_combination_list.append([learning_methods[i],[]])
# for k in range(11):
#     min_comb_and_mse = []
#     for i in range(len(learning_methods)):
#         min_comb_and_mse.append((learning_methods[i], float('inf'), ()))
#     for combination in combinations(range(11), k + 1):
#         learning_mse = calc_learning_mse(current_X, combination, attribute_df)
#         for i in range(len(min_comb_and_mse)):
#             current_learning_mse = learning_mse[min_comb_and_mse[i][0]]
#             if current_learning_mse < min_comb_and_mse[i][1]:
#                 min_comb_and_mse[i] = (min_comb_and_mse[i][0], current_learning_mse, combination)
#     print('finished '+ str(k + 1) + ' combinations')
#     for i in range(len(min_comb_and_mse)):
#         tup = min_comb_and_mse[i]
#         min_mse = tup[1]
#         min_combination = tup[2]
#         min_combination = open_tuple_with_dict(attributes_dict, min_combination)
#         learning_method_combination_list[learning_methods.index(tup[0])][1].append(Combination(min_combination, min_mse))

# %% [code]
# %matplotlib inline
# fig, axs = plt.subplots(2, 2, figsize=(25, 15))
# # quality:
# attribute = 'quality'
# attribute_num = 11
# for i in range(len(learning_methods)):
#     xs = [j + 1 for j in range(11)]
#     ys = [comb.get_mse() for comb in learning_method_combination_list[i][1]]
#     axs[int(i / 2), int(i % 2)].set_xlabel('number of attributes', fontsize=16)
#     axs[int(i / 2), int(i % 2)].set_ylabel('min mse', fontsize=16)
#     #     axs[int(attribute / 2), int(attribute % 2)].axhline(0, lw=0.5, color='black')
#     #     axs[int(attribute / 2), int(attribute % 2)].axvline(0, lw=0.5, color='black')
#     axs[int(i / 2), int(i % 2)].set_title(attribute + ' ' +learning_method_combination_list[i][0], fontsize=24)
#     axs[int(i / 2), int(i % 2)].plot(xs, ys)
#     fig.tight_layout(pad=4.0)