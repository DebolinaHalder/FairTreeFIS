#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import math
from keras import backend as K
from sklearn.tree import DecisionTreeRegressor
from scipy.special import expit
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from FairTreeFIS import fis_tree, fis_forest, fis_boosting
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor, RandomForestRegressor
from FairTreeFIS import util
import random
from sklearn.preprocessing import MinMaxScaler
import keras
from scipy.special import expit
from sklearn.model_selection import train_test_split
from scipy.special import logit, expit
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import max_norm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping

#%%
def bias_reg(x,y,pred,protected_attribute,protected_val):
    protected = ([(x,l) for (x,l) in zip(x,pred) if
        x[protected_attribute] == protected_val])
    el = ([(x,l) for (x,l) in zip(x,pred) if
        x[protected_attribute] != protected_val])
    p = sum(l for (x,l) in protected)
    q = sum(l for (x,l) in el)
    return abs(p/len(protected) - q/len(el))
    
# %%
dat = pd.read_csv("cc.csv")
dat = dat.dropna()
# %%
nrow = dat.shape[0]
target_sensitive_attribute = "racePctWhite"
#sensitive_attribute2 = "race"
target = "ViolentCrimesPerPop"
#biased_feature1 = "PctKids2Par"
#biased_feature2 = "PctLargHouseFam"
drop_features = [target_sensitive_attribute,target]

#%%
column_names = list(dat.columns)
column_names.remove(target_sensitive_attribute)
#column_names.remove(sensitive_attribute2)
column_names.remove(target)
#column_names.remove(biased_feature1)
#column_names.remove(biased_feature2)
ncol = len(column_names)

###### Visualization of Surrogate
#%%
seeds = np.arange(10)
dp = []
eq = []
accuracy = []
for i in seeds:
    keras.utils.set_random_seed(int(i))
    train, test = train_test_split(dat, test_size=0.3 ,shuffle=True,random_state = 0)
    a_train = train[target_sensitive_attribute].to_numpy()
    y_train = train[target].to_numpy()
    y_train = np.where(y_train != 1, 0, y_train)
    train = train.drop(drop_features, axis = 1)
    train = train.to_numpy()

    a_test = test[target_sensitive_attribute].to_numpy()
    y_test = test[target].to_numpy()
    y_test = np.where(y_test != 1, 0, y_test)
    test = test.drop(drop_features, axis = 1)
    test = test.to_numpy()
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(train)
    X_test = min_max_scaler.transform(test)
    
    model = Sequential()
    model.add(Dense(20, input_shape=(X_train.shape[1],), activation='relu')) # (features,)
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear')) # output node
    model.summary() # see what your model looks like

    # compile the model
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
    # early stopping callback
    es = EarlyStopping(monitor='val_loss',
                    mode='min',
                    patience=50,
                    restore_best_weights = True)
    
    model.fit(X_train, y_train,
                    validation_data = (X_test, y_test),
                    callbacks=[es],
                    epochs=100,
                    batch_size=10,
                    verbose=1)
    pred = model.predict(X_test)
    test_with_prot = np.concatenate((test,a_test.reshape(-1,1)),1)
    accuracy.append(mean_squared_error(pred,y_test))
    dp.append(abs(bias_reg(test_with_prot,y_test,pred,test.shape[1],0)))
    eq.append(abs(util.eqop(test_with_prot,y_test,pred,test.shape[1],0))) 
    
       
print(np.mean(dp),np.mean(eq),np.mean(accuracy))
#%%

##### Visualization of Random Forest
# %%

seeds = np.arange(10)
dp = []
eq = []
accuracy = []
for i in seeds:
    
    train, test = train_test_split(dat, test_size=0.3 ,shuffle=True,random_state = 0)
    a_train = train[target_sensitive_attribute].to_numpy()
    y_train = train[target].to_numpy()
    y_train = np.where(y_train != 1, 0, y_train)
    train = train.drop(drop_features, axis = 1)
    train = train.to_numpy()

    a_test = test[target_sensitive_attribute].to_numpy()
    y_test = test[target].to_numpy()
    y_test = np.where(y_test != 1, 0, y_test)
    test = test.drop(drop_features, axis = 1)
    test = test.to_numpy()
    
    clf = RandomForestRegressor(n_estimators=100)
    clf.fit(train,y_train)
    
    pred = clf.predict(test)
    test_with_prot = np.concatenate((test,a_test.reshape(-1,1)),1)
    accuracy.append(mean_squared_error(pred,y_test))
    dp.append(abs(bias_reg(test_with_prot,y_test,pred,test.shape[1],0)))
    eq.append(abs(util.eqop(test_with_prot,y_test,pred,test.shape[1],0))) 
    
       
print(np.mean(dp),np.mean(eq),np.mean(accuracy))
