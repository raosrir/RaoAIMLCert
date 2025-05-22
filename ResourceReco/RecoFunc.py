#!/usr/bin/env python
# coding: utf-8

# # What would be right Cluster Shape and Config?
# 
# The biggest question facing a customer buying DBaaS resources on the cloud is the resources and capacity of the cluster they need to buy to store their persistant data and the resources needed to process them at an acceptable SLA.
# 
# This model attempts to use information from existing operational cluster to provide recommendation on the size and shape of the cluster given the volume of data a customer has. 

# In[2]:


import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer, make_column_selector, TransformedTargetRegressor, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.utils import shuffle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import streamlit as st
import traceback

from sklearn.ensemble import VotingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, Flatten,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical, save_img, load_img, img_to_array
from tensorflow.keras.ops import expand_dims
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam, RMSprop
import keras_tuner

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier


# In[ ]:





# # Generic Model Engine

# ## Split Data into Train and Test 
# 
# We will use an **80-20 train-test split** to train our model and then test the model for it's efficiency. Further we will randomize it as well to disperse any unintended order of data in the source. 

# ## Modeling
# 
# We will now employ different models for supervised learning and measure their efficacy before picking one of them.

# ## Regression Models

# In[9]:


# a function that takes the best estimator and computes feature importance and displays them

def compute_importance (best_est, X_test, y_test, repeats, num_cols=5):
    imp = permutation_importance(best_est, X_test, y_test, n_repeats=repeats, random_state=0)
    
    # Print model name
    print('Model Name')
    print('----------')
    print(best_est)
    
    print('')
    # Print feature importances
    print('Feature Importances')
    print('-------------------')
    # print the importance means
    for i in imp.importances_mean.argsort()[::-1]:
            print(f"{X_test.columns[i]:<8} "
               f"{imp.importances_mean[i]:.3f}"
               f" +/- {imp.importances_std[i]:.3f}")
    
    # print estimator score and MSE
    print(' ')
    score = best_est.score(X_test, y_test)
    print(f'Estimator Score:', score)
    mse = mean_squared_error(y_test, best_est.predict(X_test))
    print(f'Prediction Error (MSE)', mse)
    
    # also display it visually
    feature_importances = pd.Series(imp.importances_mean, index=X_test.columns).sort_values(ascending=False)
    feature_importances = feature_importances.iloc[:num_cols]
    
    fig, ax = plt.subplots()
    feature_importances.plot.bar()
    
    ax.set_title("Importance of Features in Determining Used Car Sale Price")
    ax.set_ylabel("Importance Mean")
    ax.set_xlabel("Cluster Features")
    fig.tight_layout()
    
    return score, mse


# ### Linear Regression with Grid Search on Ploynomial Features

# In[12]:


# Linear Regression

def linear_regression(dev_train, target_train, dev_test, target_test, col_trans):
    
    pipe = Pipeline([   ("transformer", col_trans),
                        ("poly", PolynomialFeatures(include_bias=False)),
                        ("linreg", LinearRegression())])

    # Use grid search to find optimal poly degree
    param_dict = {'poly__degree': [1, 2, 3, 4]}
    grid_lin = GridSearchCV(estimator=pipe, param_grid=param_dict)
    grid_lin.fit(dev_train, target_train)
    
    score, mse = compute_importance(grid_lin.best_estimator_, dev_test, target_test, 30)
    return score, mse, grid_lin.best_estimator_



# ### Sequential Feature Selection with Grid Search for Optimal number of features

# In[17]:


def seq_feature_select(dev_train, target_train, dev_test, target_test, col_trans):
    selector_pipe = Pipeline([  ('transformer', col_trans),
                        ('selector', SequentialFeatureSelector(LinearRegression())),
                        ('model', LinearRegression())])

    param_dict = {"selector__n_features_to_select":[2,3,4,5]}
    selector_grid = GridSearchCV(estimator=selector_pipe, param_grid=param_dict)

    selector_grid.fit(dev_train, target_train)

    best_estimator = selector_grid.best_estimator_
    score, mse = compute_importance(best_estimator, dev_test, target_test, 30)
    
    return score, mse, best_estimator


# ### Ridge with Grid Search to choose optimal Alpha

# In[21]:


def ridge_grid (dev_train, target_train, dev_test, target_test, col_trans):    
    ridge_pipe = Pipeline([
                        ("transformer", col_trans),
                        ('Ridge', Ridge())])
    params_dict = {"Ridge__alpha":[1e-10, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]}

    ridge_grid = GridSearchCV(ridge_pipe, param_grid=params_dict)
    ridge_grid.fit(dev_train, target_train)

    best_ridge = ridge_grid.best_estimator_.named_steps['Ridge']

    trans_pipe = Pipeline([ # ('transformer', col_transformer), 
                    ('regressor', TransformedTargetRegressor(regressor=best_ridge))])
    trans_pipe.fit(dev_train, target_train)

    score, mse = compute_importance(trans_pipe, dev_test, target_test, 30)
    return score, mse, trans_pipe


# ### RFE with Lasso Estimator

# In[25]:


def rfe_lasso(dev_train, target_train, dev_test, target_test, col_trans):
    
    rfe_pipe = Pipeline([ ('transformer', col_trans),
                        ('rfe', RFE(estimator=Lasso(), n_features_to_select=8))])

    param_dict = {"rfe__n_features_to_select":[2,3,4,5,6,7,8]}
    rfe_grid = GridSearchCV(estimator=rfe_pipe, param_grid=param_dict)

    rfe_grid.fit(dev_train, target_train)

    best_estimator = rfe_grid.best_estimator_
    score, mse = compute_importance(best_estimator, dev_test, target_test, 30)
    return score, mse, best_estimator


# ## Ensemble Models

# ### Voting Regressor with GridSearch to choose weights

# In[ ]:


# Voting Regressor

def voting_reg(dev_train, target_train, dev_test, target_test, col_trans):
    
    vr = VotingRegressor(estimators=[
                                        ('lgtr', TransformedTargetRegressor(regressor=LinearRegression())), 
                                          ('knr', KNeighborsRegressor()),
                                         ('dtr', DecisionTreeRegressor()),
                                          ('rid', Ridge()),
                                          ('svr', SVR())
                                    ])
    vr_pipe = Pipeline([  ('transformer', col_trans), 
                            ('vr', vr)])

    param_dict = {"vr__weights":[[0.25, 0.10, 0.25, 0.25, 0.10],
                                 [0.25, 0.25, 0.25, 0.10, 0.10],
                                 [0.25, 0.10, 0.10, 0.25, 0.25]]}
    vr_grid = GridSearchCV(estimator=vr_pipe, param_grid=param_dict)

    mult = MultiOutputRegressor(vr_grid)
    mult.fit(dev_train, target_train)

    # best_estimator = mult.estimators_[0].best_estimator_
    score, mse = compute_importance(mult, dev_test, target_test, 30)
    return score, mse, mult


# ### Random Forest with GridSearch to pick estimators and depth

# In[ ]:


# Random Forest

def random_forest(dev_train, target_train, dev_test, target_test, col_trans, boost=False):
    rfr_pipe = Pipeline([ ('transformer', col_trans), 
                         ('rfr', RandomForestRegressor())])

    param_dict = {'rfr__n_estimators':[1, 10, 100, 500, 1000, 2000],
                 'rfr__max_depth':[1, 2, 3, 4, 5, None]}
    rfr_grid = GridSearchCV(estimator=rfr_pipe, param_grid=param_dict)
    rfr_grid.fit(dev_train, target_train)
    
    best_estimator = rfr_grid.best_estimator_
    
    score, mse = compute_importance(best_estimator, dev_test, target_test, 10)
    if (boost):
        ret_val = score, mse, best_estimator, rfr_grid.best_params_['rfr__max_depth'], rfr_grid.best_params_['rfr__n_estimators']
    else: 
        ret_val = score, mse, best_estimator

    return ret_val


# ### Boosting

# In[35]:


def random_forest_boost(dev_train, target_train, dev_test, target_test, col_trans):
    
    score, mse, model, max_depth, n_estimators = random_forest(dev_train, target_train, 
                                                dev_test, target_test, col_trans, True)
    
    abr = AdaBoostRegressor(estimator=RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth), 
                                random_state=42)

    abr_pipe = Pipeline([ # ('transformer', col_trans), 
                         ('abr', abr)])
    # abr_pipe.fit(dev_train, target_train)

    # compute_importance(abr_pipe, dev_test, target_test, 3)

    chain = RegressorChain(abr_pipe)
    chain.fit(dev_train, target_train)

    score, mse = compute_importance(chain, dev_test, target_test, 3)
    return score, mse, chain


# ## Neural Network Model

# In[38]:


# Prepare data for NN

def neural_net(X, y, col_trans):
    
    Xnn = pd.concat([X, y], axis=1)
    Xc = Xnn.copy()
    Xnn = shuffle(Xnn)
    
    features = np.asarray(Xnn).astype('float32')
    train, dev = np.split(features, [int(len(features) * 0.8)])

    end = -(len(y.columns))
    train_features = train[:, :end]
    train_target = train[:, end:]

    dev_features = dev[:, :end]
    dev_target = dev[:, end:]

    cnn = Sequential()
    cnn.add(Flatten())
    cnn.add(Dense(64, activation='relu'))
    cnn.add(Dense(64, activation='relu'))
#    cnn.add(Dense(64, activation='relu'))
#    cnn.add(Dense(64, activation='relu'))
    cnn.add(Dense(64, activation='relu'))
    cnn.add(Dense(-end, activation=None))

    # cnn.compile(optimizer='rmsprop', loss='mae', metrics=['mse'])
    cnn.compile(optimizer='rmsprop', loss='mae', metrics=['mse', 'accuracy'])

    history = cnn.fit(train_features, train_target, validation_data=(dev_features, dev_target), 
                                          epochs=1000, verbose=0)
    print(f'MSE using Convolutional Neural Networks is: {history.history["mse"][-1]}')
    print(f'Accuracy using Convolutional Neural Networks is: {history.history["accuracy"][-1]}')
    
    return history.history["accuracy"][-1], history.history["mse"][-1], cnn


# In[40]:


# Neural Network with hyperparameter search

def cnn_optimal(hp, end):
    model = Sequential()
    model.add(Flatten())
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )
    if hp.Boolean("dropout"):
        model.add(Dropout(rate=0.25))
    model.add(Dense(-end, activation=None))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=RMSprop(learning_rate=learning_rate),
        loss="mae",
        metrics=["mse","accuracy"],
    )
    return model


# In[42]:


def neural_best(X, y, col_trans):
    
    Xnn = pd.concat([X, y], axis=1)
    Xc = Xnn.copy()
    Xnn = shuffle(Xnn)
    
    features = np.asarray(Xnn).astype('float32')
    train, dev = np.split(features, [int(len(features) * 0.8)])

    end = -(len(y.columns))
    train_features = train[:, :end]
    train_target = train[:, end:]

    dev_features = dev[:, :end]
    dev_target = dev[:, end:]

    tuner = keras_tuner.RandomSearch(
        hypermodel=lambda hp: cnn_optimal(hp, end),
        objective="accuracy",
        max_trials=3,
        executions_per_trial=2,
        overwrite=True,
        directory="cnn_dir",
        project_name="Reco",
    )
    tuner.search(train_features, train_target, epochs=50, validation_data=(dev_features, dev_target))
    models = tuner.get_best_models(num_models=2)
    cnn = models[0]
    
    history = cnn.fit(train_features, train_target, validation_data=(dev_features, dev_target), 
                                          epochs=1000, verbose=0)
    
    print(f'MSE using Convolutional Neural Networks is: {history.history["mse"][-1]}')
    print(f'Accuracy using Convolutional Neural Networks is: {history.history["accuracy"][-1]}')
    
    return history.history["accuracy"][-1], history.history["mse"][-1], cnn


# In[44]:


# dev_scores = cnn.evaluate(dev_features, dev_target, verbose=2)
# print("Test loss:", dev_scores[0])
# print("Test accuracy:", dev_scores[1])

# predictions = cnn.predict(train_features[:5])
# print(predictions)


# In[46]:


def best_model(X, y, col_trans):
    
    X_train, X_test, y_train, y_test = split_test_train(X, y)
    
    score, next_mse = 0, 0
    
#    for func in [linear_regression, seq_feature_select, ridge_grid, 
#                         rfe_lasso, voting_reg, random_forest, random_forest_boost]:
#        next_score, next_mse, next_model = func(X_train, y_train, X_test, y_test, col_trans)   
#        if (next_score > score):
#            score = next_score
#            model = next_model
  
    next_score, next_mse, next_model = neural_net(X, y, col_trans)
    
#    return next_model if next_score > score else model

    return next_model  # for now skip comparison with other models since we know nueral is performing best for Reco


# ## Classification Models

# In[49]:


# Decision Tree Classifier 

def decision_tree(dev_train, target_train, dev_test, target_test, col_trans):
 
    dt_mult = MultiOutputClassifier(DecisionTreeClassifier())
    params = {'estimator__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'estimator__min_samples_split': [0.05, 0.1, 0.15, 0.2],
              'estimator__criterion': ['gini','entropy','log_loss'],
              'estimator__min_samples_leaf': [0.05, 0.1, 0.15, 0.2]
             }
    dt_grid = GridSearchCV(estimator=dt_mult, param_grid=params)
    
    dt_pipe = Pipeline([
            ('coltrans', col_trans), 
            ('grid', dt_grid)])
    dt_pipe.fit(dev_train, target_train)

    score, mse = compute_importance(dt_pipe, dev_test, target_test, 30)
    return score, mse, dt_pipe


# In[51]:


def xtree_ens(dev_train, target_train, dev_test, target_test, col_trans):
 
    xtrees_mult = MultiOutputClassifier(ExtraTreesClassifier())
    params = {'estimator__n_estimators': [10, 50, 100, 500, 1000], # 1500, 1900, 2100, 3000],
              'estimator__criterion': ['gini', 'entropy'],
              'estimator__max_depth': [1, 5, 13, 34, 54, 89, None],
              'estimator__min_samples_split': [2, 8, 21, 55, 144, 377]
#              'estimator__min_samples_leaf': [1, 5, 13, 34, 89, 233, 377]
             }
    xt_grid = GridSearchCV(estimator=xtrees_mult, param_grid=params)
    
    xt_pipe = Pipeline([
            ('coltrans', col_trans), 
            ('grid', xt_grid)])
    xt_pipe.fit(dev_train, target_train)

    score, mse = compute_importance(xt_pipe, dev_test, target_test, 30)
    return score, mse, xt_pipe


# In[53]:


def knn(dev_train, target_train, dev_test, target_test, col_trans):
 
    knn = MultiOutputClassifier(KNeighborsClassifier())
    params = {'estimator__n_neighbors':np.array(range(1, 21, 2))}
    knn_grid = GridSearchCV(estimator=knn, param_grid=params)
    
    knn_pipe = Pipeline([
            ('coltrans', col_trans), 
            ('grid', knn_grid)])
    knn_pipe.fit(dev_train, target_train)

    score, mse = compute_importance(knn_pipe, dev_test, target_test, 30)
    return score, mse, knn_pipe


# In[55]:


def rand_forest_cl(dev_train, target_train, dev_test, target_test, col_trans):
 
    rfc = MultiOutputClassifier(RandomForestClassifier())

    params = {'estimator__n_estimators':[1, 10, 100, 500, 1000, 2000],
                 'estimator__max_depth':[1, 2, 3, 4, 5, None]}
    rfc_grid = GridSearchCV(estimator=rfc, param_grid=params)
    
    rfc_pipe = Pipeline([
            ('coltrans', col_trans), 
            ('grid', rfc_grid)])
    rfc_pipe.fit(dev_train, target_train)

    score, mse = compute_importance(rfc_pipe, dev_test, target_test, 30)
    return score, mse, rfc_pipe


# In[57]:


def xtree(dev_train, target_train, dev_test, target_test, col_trans):
 
    xtrees_mult = MultiOutputClassifier(ExtraTreeClassifier())
    params = {'estimator__max_depth': [1, 5, 13, 34, 54, 89, None],
              'estimator__min_samples_split': [2, 8, 21, 55, 144, 377],
              'estimator__min_samples_leaf': [1, 5, 13, 34, 89, 233, 377]
             }
    xt_grid = GridSearchCV(estimator=xtrees_mult, param_grid=params)
    
    xt_pipe = Pipeline([
            ('coltrans', col_trans), 
            ('grid', xt_grid)])
    xt_pipe.fit(dev_train, target_train)

    score, mse = compute_importance(xt_pipe, dev_test, target_test, 30)
    return score, mse, xt_pipe


# In[59]:


def best_class(X, y, col_trans):
    
    X_train, X_test, y_train, y_test = split_test_train(X, y)
    
    score, next_mse = 0, 0
    for func in [decision_tree, xtree_ens, knn, rand_forest_cl, xtree]:
        next_score, next_mse, next_model = func(X_train, y_train, X_test, y_test, col_trans)   
        if (next_score > score):
            score = next_score
            model = next_model
    
    return next_model if next_score > score else model


# In[61]:


def split_test_train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# # Load data

# In[64]:


def clean_node_data(nodes):
    # strip trailing % and GB chars from nodes

    nodes['cpuUtil'] = nodes['CPU Utilization'].str.slice(stop=-1).astype(float)
    nodes['ramUtil'] = nodes['RAM Utilization'].str.slice(stop=-1).astype(float)
    nodes['diskUtil'] = nodes['Disk Utilization'].str.slice(stop=-1).astype(float)
    nodes['dataSize'] = nodes['Disk Used'].str.slice(stop=-3).astype(float)
    nodes = nodes.drop(['CPU Utilization', 'RAM Utilization', 'Disk Utilization', 'Disk Used'], axis=1)

    # extract services from Node Name
    nodes['services'] = nodes['Node Name'].str.split("-", expand=True)[1]
    
    # fill dataSize NaNs with 0
    nodes['dataSize'] = nodes['dataSize'].fillna(0)
    
    nodes['servcs_num'] = LabelEncoder().fit_transform(nodes['services'])
    
    return nodes


# In[66]:


# replace missing values of iops with the most used iops value

def replace_missing_iops(clusters_raw):
    
    iops_mode_lst = clusters_raw['iops'].mode()[0]
    # print(type(iops_mode_lst))
    iops_mode = int(sum(iops_mode_lst) / len(iops_mode_lst))

    nan_locs = clusters_raw.query('iops.isna()').index
    for loc in nan_locs:
        iops = []
        for i in range(len(clusters_raw.loc[loc, 'nodes'])):
            iops.append(iops_mode)
        clusters_raw.loc[loc, 'iops'] = iops
        
    return clusters_raw


# In[68]:


@st.cache_data
def load_data():
    # load clusters data and fill missing values
    clusters_raw = pd.read_json('clusterShape.json')
    clusters_raw = replace_missing_iops(clusters_raw)
    
    # load node data and clean it up
    nodes = pd.read_csv('nodes.csv')
    nodes = clean_node_data(nodes)
    
    return clusters_raw, nodes


# # Feature Engineering

# ## Data Inspection and Cleanup

# ### Work on the Nodes dataset

# In[74]:


# Max service groups

def max_groups(clusters_raw):
    max_grps = 1
    for val in clusters_raw['nodes']:
        max_grps = len(val) if len(val) > max_grps else max_grps
    return max_grps


# ### Work on the Clusters dataset

# In[77]:


# Flatten and get unique values

def flat_unique(col):
    flat = [val
            for vals in col
            for val in vals]
    lst = list(set(flat))
    lst.sort()
    return lst

@st.cache_data
def feature_vals(clusters_raw):
    feature_vals = {}
    # Unique values for Cols
    for col in ['services', 'nodes', 'cpu', 'RAM', 'cpuType', 'storage', 'diskType', 'iops']:
        feature_vals[col] = flat_unique(clusters_raw[col].dropna())
    
    return feature_vals

def flat_tot(col):
    tot = [sum(vals)
            for vals in col]
    lst = list(set(tot))
    lst.sort()
    return lst

@st.cache_data
def feature_tots(clusters_raw):
    feature_tot = {}
    # Total values for Cols
    for col in ['nodes', 'cpu', 'RAM', 'storage']:
        feature_tot[col] = flat_tot(clusters_raw[col].dropna())

    return feature_tot


# In[79]:


def find_nearest(array, value):
    array = np.asanyarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

    
def nearest_res(configs, feature_vals, conf=False):
    near_vals = []
    
    vals = 4 if (conf) else 1
    clusters = 0
    i = 0
    
    # nearest node value
    # add a 0 to the list for node config to indicate zero allocation to a node
    for j in range(vals):
        near_vals.append(find_nearest(feature_vals['nodes'] + [0] if (conf) else feature_vals['nodes'], 
                                      configs[0][i]))
        
        # keep count of number of clusters
        if (conf and near_vals[j] > 0):
            clusters = clusters + 1
        print(configs[0][i])
        i = i + 1
    
    # nearest CPU value
    for j in range(vals):
        if (conf and j >= clusters):
            near_vals.append(0) # no more clusters - ignore CPU predictions
        else:
            near_vals.append(find_nearest(feature_vals['cpu'] + [0] if (conf) else feature_vals['cpu'], 
                                      configs[0][i]))
        print(configs[0][i])
        i = i + 1
    
    # nearest RAM value
    for j in range(vals):
        if (conf and j >= clusters):
            near_vals.append(0) # no more clusters - ignore RAM predictions
        else:
            near_vals.append(find_nearest(feature_vals['RAM'] + [0] if (conf) else feature_vals['RAM'], 
                                      configs[0][i]))
        print(configs[0][i])
        i = i + 1
    
    # nearest disk value
    for j in range(vals):
        if (conf and j >= clusters):
            near_vals.append(0) # no more clusters - ignore disk predictions
        else:
            near_vals.append(find_nearest(feature_vals['storage'] + [0] if (conf) else feature_vals['storage'], 
                                      configs[0][i]))
        print(configs[0][i])
        i = i + 1
    
    # nearest iops value
    if (conf):
        for j in range(vals):
            if (j >= clusters):
                near_vals.append(0) # no more clusters - ignore iops predictions
            else:
                near_vals.append(find_nearest(feature_vals['iops'] + [0] if (conf) else feature_vals['iops'], 
                                      configs[0][i]))
        print(configs[0][i])
        i = i + 1

    return near_vals

def match_con_totals (con_est, cluster_est):
    
    i = 0
    for val in cluster_est:
            
            # get the correct value list 
            j = 0
            val_list = []
            for col in ['nodes', 'cpu', 'RAM', 'storage', 'diskType']:
                val_list = feature_vals[col]
                if (j >= i):
                    break
                j = j + 1
            
            tot = 0
            while tot < val:
                for k in range(4):
                    if (con_est[i+k] != 0):
                        con_est[i+k] = find_next_highest(con_est[i+k], val_list)
                        tot = tot + con_est[i+k]
                        


# # Create dataframes for the models

# ## Create Cluster Dataframe
# 
# Merge the cluster and node data into one dataframe based on database id

# In[83]:


services = ['cbas', 'eventing', 'fts', 'index', 'kv', 'n1ql']
cluster_target = ['nodes', 'cpu', 'ram', 'disk']

def create_cluster():
    return pd.DataFrame(columns=services +['dataSize',
                            'cpuUtil', 'ramUtil', 'diskUtil']
                            + cluster_target)


# In[85]:


def create_config_cols(max_groups):
    columns = []
    
    for cols in ['svcs'] + cluster_target + ['iops']:
        for i in range(max_groups):
            columns.append(cols + "_" + str(i))
            
    return columns


# In[87]:


# Merge Cluster and Node data

def insert_service(srvcs, includeKv=True):
    new_row = []
    
    # insert service
    for service in services:
        if (service != 'kv' or (service == 'kv' and includeKv)):
            if service in srvcs:
                new_row.append(1)
            else:
                new_row.append(0)
            
    return new_row

def merge_cluster_node(clusters_raw, nodes, clusters):
    loc = 0
    for index, row in clusters_raw.iterrows():

        # call utility method to insert services
        new_row = insert_service(row['services'])

        # insert data size - sum of data size on each cluster node
        new_row.append(nodes[nodes['Database ID']==row['id']]['dataSize'].sum())

        # insert metrics
        new_row.append(nodes[nodes['Database ID']==row['id']]['cpuUtil'].max())
        new_row.append(nodes[nodes['Database ID']==row['id']]['ramUtil'].max())
        new_row.append(nodes[nodes['Database ID']==row['id']]['diskUtil'].max())

        # insert configs
        new_row.append(np.array(row['nodes']).sum())
        new_row.append(np.array(row['cpu']).sum())
        new_row.append(np.array(row['RAM']).sum())
        new_row.append(np.array(row['storage']).sum())

        # insert new row into df
        clusters.loc[loc] = new_row
        loc+=1
        
    return clusters


# In[89]:


# Fill null values with mean

def fill_missing_cl_values(clusters, nodes):
    clusters['cpuUtil'] = clusters['cpuUtil'].fillna(nodes['cpuUtil'].mean())
    clusters['ramUtil'] = clusters['ramUtil'].fillna(nodes['ramUtil'].mean())
    clusters['diskUtil'] = clusters['diskUtil'].fillna(nodes['diskUtil'].mean())

    # insert approximation for data size (we need to get the actual number for this from the source)
    # clusters['dataSize'] = (clusters['diskUtil'] / 100 * clusters['disk']) * 0.9
    
    return clusters


# In[91]:


@st.cache_data
def create_cluster_df(clusters_raw, nodes):
    
    clusters = create_cluster()
    clusters = merge_cluster_node(clusters_raw, nodes, clusters)
    clusters = fill_missing_cl_values(clusters, nodes)
    
    return clusters


# ### Inspect Data Range and Values for feature selection

# In[94]:


# Get minimum and max values for relevant columns for inspection

def min_max(col):
    minMax = [clusters[col].min(), clusters[col].max()]
    return minMax


# ### Features we will not use
# 
# To mitigate the 'Curse of Dimensionality' we will do away with feature that we do not expect to contribute to our model. We will drop the following feature from the model for the reasons cited.
# 
# 1. **kv**: There are no clusters without a KV service installed and hence we'll drop that feature

# In[98]:


@st.cache_data
def create_cluster_X_y(clusters):
    df = pd.DataFrame(columns=['cbas', 'eventing', 'fts', 'index', 'n1ql', 'dataSize',
                            'cpuUtil', 'ramUtil', 'diskUtil'])
    X_cl = clusters[df.columns]
    y_cl = clusters[cluster_target]
    
    return X_cl, y_cl


# ### Preprocessing

# ### Train model on Cluster dataframe

# In[102]:


@st.cache_resource
def train_cluster(X, y):
    col_trans_cl = make_column_transformer(
                    (StandardScaler(), X.select_dtypes(include=np.number).columns.tolist()),
                    remainder='passthrough')
    return best_model(X, y, col_trans_cl)


# ## Create Config dataframe

# In[105]:


@st.cache_data
def create_conf_X_y(clusters, nodes, clusters_raw):
    
    max_grps = max_groups(clusters_raw)
    
    # start with the clusters data
    configs = clusters.copy()

    # drop the utils cols
    configs = configs.drop(['cpuUtil', 'ramUtil', 'diskUtil'], axis=1)

    # numeric to represent empty string for services
    svc_none = nodes['servcs_num'].max() + 1

    # add other columns and initialize
    for col in create_config_cols(max_grps):
        if col.startswith('svcs'):
            configs[col] = svc_none # number to represent empty string 
        else:
            configs[col] = 0

    groups_count = max_grps
    loc = 0
    for index, row in clusters_raw.iterrows():

        svcs = nodes[nodes['Database ID']==row['id']]['servcs_num'].tolist()

        # move to next cluster if no node info for this cluster
    #    if (len(svcs) < 1):
    #        continue

        # create unique value list
        svc_dedup = []
        _ = [svc_dedup.append(x) for x in svcs if x not in svc_dedup]

        # set services for each node group
        i = 0
        for k in row['nodes']:
            for l in svc_dedup:
                if svcs.count(l) == k: 
                    configs.loc[loc, 'svcs_' + str(i)] = l
                    i+=1
                    svc_dedup.remove(l)
                    break

        i = 0
        # set nodes for each node group
        for node in row['nodes']:
            configs.loc[loc, 'nodes_' + str(i)] = int(node)
            i+=1

        i = 0
        # set cpu for each node group
        for cpu in row['cpu']:
            configs.loc[loc, 'cpu_' + str(i)] = int(cpu)
            i+=1

        i = 0
        # set RAM for each node group
        for ram in row['RAM']:
            configs.loc[loc, 'ram_' + str(i)] = int(ram)
            i+=1

        i = 0
        # set disk for each node group
        for disk in row['storage']:
            configs.loc[loc, 'disk_' + str(i)] = int(disk)
            i+=1

        i = 0
        # set iops for each node group
        for iops in row['iops']:
            configs.loc[loc, 'iops_' + str(i)] = int(iops)
            i+=1

        loc+=1
        
    X_con = configs.drop(create_config_cols(max_grps) + ['kv'], axis=1)
    y_con = configs.drop(X_con.columns, axis=1)

    # for now we are not predicting service placement in the nodes
    y_con = y_con.drop(['kv', 'svcs_0', 'svcs_1', 'svcs_2', 'svcs_3'], axis=1)

    return X_con, y_con


# ### Train model on Configs dataframe

# In[108]:


@st.cache_resource
def train_config(X, y):
    cols = ['nodes', 'cpu', 'ram', 'disk', 'iops']
    models = []
    
    for each in cols:
        y_each = y[[each + '_0', each + '_1', each + '_2', each + '_3']].copy()
        col_trans_con = make_column_transformer(
                    (StandardScaler(), X.select_dtypes(include=np.number).columns.tolist()),
                    remainder='passthrough')
        models.append(best_model(X, y_each, col_trans_con))
        
    return models


# In[110]:


# col_trans_con = make_column_transformer(
#                (StandardScaler(), X_con.select_dtypes(include=np.number).columns.tolist()),
#                remainder='passthrough')
# con_acc, con_mse, best_model_con = neural_best(X_con, y_con, col_trans_con)


# # The Resource Estimator 
# 
# Now we will use our best model to build a resource estimator which a cloud buyer can use to estimate the resources he will need to run particular services on Couchbase given the volume of data that will be stored.
# 
# The model with best score (and lowest MSE) is Neural Network which beats other models with a impressive accuracy score of 98+% and a low 4 digit MSE.
# 
# This model will then be hosted on a website for easy access and use. 
# 
# Below is a trial run.

# In[113]:


# Function to estimate the resources namely number of nodes, cpus, size of RAM and disk given the services needed,
# size of user data, desired CPU, RAM and disk utilization

def resource_estimate(best_model_cl, services, data_size, cpuUtil, ramUtil, diskUtil):
    
    X_row = insert_service(services, False)

    X_row.append(data_size)
    X_row.append(cpuUtil)
    X_row.append(ramUtil)
    X_row.append(diskUtil)
    
    if ('sequential' in best_model_cl.name):
        features = np.asarray(X_row).astype('float32')
        features = features.reshape(-1, len(X_row))
    else:
        features = [X_row]
    
    # Predic price for the particular with the input features 
    resource_estimate = best_model_cl.predict(features)
    
    return resource_estimate


# In[115]:


def config_estimate(models, services, data_size, resource_est):
    
    X_row = insert_service(services, False)
    
    X_row.append(data_size)
    for val in resource_est:
        X_row.append(val)
    
    con_estimates = []
    for model in models:
        print(model)
        if ('sequential' in model.name):
            features = np.asarray(X_row).astype('float32')
            features = features.reshape(-1, len(X_row))
        else:
            features = [X_row]

        # Predic price for the particular with the input features 
        con_estimates.append(model.predict(features))
    
    return con_estimates


# # Conclusion
# 
# We now a have resource estimator tool that uses historical production server preformance data to provide a Capella user to buy the right resources (Nodes, CPU, Memory and Disk) based on their data foot print and system usage tolerance.
# 
# The tool developed gives the customer the ability to 'right size' their capital expenditure trailored to there needs there by preventing a system where resources are underutilized or one that doesn't have enough resources to run their workloads in a performant manner.

# In[118]:


def load_data_and_train(services, data_size, cpuUtil, ramUtil, diskUtil):
    
    # load data
    clusters_raw, nodes = load_data()

    # train on cluster resources
    clusters = create_cluster_df(clusters_raw, nodes)
    X_cl, y_cl = create_cluster_X_y(clusters)
    best_model_cl = train_cluster(X_cl, y_cl)

    # train on cluster configuration
    X_con, y_con = create_conf_X_y(clusters, nodes, clusters_raw)
    best_model_con = train_config(X_con, y_con)
    
    # collect unique feature values from data
    feature_totals = feature_tots(clusters_raw)

    # estimate resources based on user input
    res_est = resource_estimate(best_model_cl, services, data_size, cpuUtil, ramUtil, diskUtil)
    res_est = nearest_res(res_est, feature_totals)
    print (f'Resource Estimates: {res_est}')

    # collect unique feature values from data
    feature_values = feature_vals(clusters_raw)

    # estimate configuration for the cluster
    conf_est = config_estimate(best_model_con, services, data_size, res_est)
    conf_est = nearest_res(conf_est, feature_values, True)
    print (f'Configuration Estimate: {conf_est}')
    
    return res_est

def display_result_form(res_est):

    # Display a title
    st.title("Recommended Configuration")

    # Add other content below the title
    st.write("This is the suggested configuration based on the requirement above")

    # Use columns to display them in one row
    nodes, cpus, ram, disk = st.columns(4)

    # Add labels and values to each column
    nodes.markdown("Nodes")
    nodes.write(res_est[0])

    cpus.markdown("CPUs")
    cpus.write(res_est[1])

    ram.markdown("RAM (GB)")
    ram.write(res_est[2])

    disk.markdown("Disk (GB)")
    disk.write(res_est[3])
    
def display_input_form():
    
    # Row 1: Six checkboxes, with the first one selected and disabled
    row1 = st.columns([0.2, 0.09, 0.13, 0.13, 0.13, 0.15, 0.15])
    row1[0].subheader('Pick Services Needed')
    check_kv = row1[1].checkbox('KV', value=True, disabled=True)
    check_query = row1[2].checkbox('Query')
    check_index = row1[3].checkbox('Index')
    check_fts = row1[4].checkbox('Text Search')
    check_event = row1[5].checkbox('Eventing')
    check_cbas = row1[6].checkbox('Analytics')

    # Row 2: Integer text box
    st.subheader('Data Size (in GB)')
    data_size = st.number_input('Enter an integer:', min_value=1, step=5)

    # Row 3, 4, 5: Sliders for percentage selection (25% to 90%)
    st.subheader('Target CPU Utilization')
    slider_cpu = st.slider('Select Percentage (25% to 90%)', min_value=25, max_value=90, step=5, value=50, key=1)

    st.subheader('Target Memory Utilization')
    slider_ram = st.slider('Select Percentage (25% to 90%)', min_value=25, max_value=90, step=5, value=50, key=2)

    st.subheader('Target Disk Utilization %')
    slider_disk = st.slider('Select Percentage (25% to 90%)', min_value=25, max_value=90, step=5, value=50, key=3)

    # Submit button at the bottom
    st.subheader('Submit the Form')
    submit_button = st.button('Submit')

    if submit_button:

        # Capture input values
        services = []
        if (check_cbas):
            services.append('cbas')
        if (check_event):
            services.append('eventing')
        if (check_fts):
            services.append('fts')
        if (check_index):
            services.append('index')
        if (check_query):
            services.append('query')
        
        res_est = load_data_and_train(services, data_size, slider_cpu, slider_ram, slider_disk)
        
        display_result_form(res_est)
        


# Call the function to display the form
# display_form()


# In[120]:


def main():

    # display form for input
    display_input_form()


if __name__ == "__main__":
    main()


# In[ ]:




