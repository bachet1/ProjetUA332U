# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:52:47 2020


@author: luc

"""
# Modélisation des medeles de machines learning


# 1 - Importation des librairies de machine learning


import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV
from scipy import stats 
import seaborn as sns

# 2 - Importation des données
data = pd.read_csv("https://raw.githubusercontent.com/oreilly-mlsec/book-resources/599669c7124dffb65ea7f6e0b7626df32496b1d6/chapter3/datasets/cpu-utilization/cpu-full-a.csv",sep=",")

data_total = pd.read_csv("https://raw.githubusercontent.com/oreilly-mlsec/book-resources/599669c7124dffb65ea7f6e0b7626df32496b1d6/chapter3/datasets/cpu-utilization/cpu-full-a.csv",parse_dates=[0], index_col=0)

# 3 - séparation des données d'apprentissage et de test


train_size=int(len(data_total)*.80)

test_size=len(data_total)-train_size

train,test=data_total.iloc[0:train_size],data_total.iloc[train_size:len(data_total)]



# 4 - Traitement des données et créations des features y

# 4.1 : Création d'une fonction pour traiter les données

def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['minute'] = df['date'].dt.minute
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['minute','hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X


train_x,y_train= create_features(train,label='cpu')
test_x, y_test = create_features( test, label='cpu')
x_total, y_total = create_features(data_total, label='cpu')



# 5 - Application des algorithmes de machine learning sur les données


modelxgb = xgb.XGBRegressor(objective ='reg:linear',n_estimators=100)
modelxgb.fit(train_x, y_train,xgb_model=None, eval_set=[(train_x, y_train), (test_x, y_test)],
        early_stopping_rounds=100, verbose=0) 

modelAbaB = AdaBoostRegressor(random_state=0, n_estimators=100)
modelAbaB.fit(train_x, y_train)

modelRandomF=RandomForestRegressor(max_depth=2, random_state=0)
modelRandomF.fit(train_x, y_train)

modelSVR=SVR(C=1.0, epsilon=0.2)
modelSVR.fit(train_x, y_train)


# 6 - Calcul de r2_score par technique

print(" la precision du modèle xgb est : ",r2_score(y_train,modelxgb.predict(train_x)) )                                      
print("la precision du modèle Adaboost : ",r2_score(y_train,modelAbaB.predict(train_x)))
print("la precision du modèle Random Forest ", r2_score(y_train,modelRandomF.predict(train_x)))
print("la precision du modèle SVR ",r2_score(y_train,modelSVR.predict(train_x))) 


# 7 -  Prediction des données Test 

# 7.1 : Prédiction sur les données  Test 

predTestXGB= modelxgb.predict(test_x)
predTestAdab=modelAbaB.predict(test_x)
predTestRandf=modelRandomF.predict(test_x)
predTestSvr=modelSVR.predict(test_x)


# visualisation for data
    
#model for xgboost
test['cpu_Predictionxgb'] = predTestXGB
# modèle AbaBoost
test['cpu_PredictionAdab'] = predTestAdab
# modèle Random Forest
test['cpu_PredictionRandomF'] =predTestRandf
# modèle Random Forest
test['cpu_PredictionSVR'] = predTestSvr


_ = test[['cpu','cpu_Predictionxgb','cpu_PredictionAdab','cpu_PredictionRandomF','cpu_PredictionSVR']].plot(figsize=(15, 5))




# 7.3 - Calcul des mean squarre et erreurs

print("MSE modèle xgb test ",mean_squared_error(y_test,predTestXGB))                                       
print("MSE modèle Adaboost test ",mean_squared_error(y_test,predTestAdab))
print("MSE modèle Random Forest test ", mean_squared_error(y_test,predTestRandf))
print("MSE modèle SVR test ",mean_squared_error(y_test,predTestSvr)) 




# 8 - Optimisation des paramettres


# 8.1 : parametres RandomForet

dico_paramrf = {
    'bootstrap': [True],
    'max_depth': [5,10,20],
    'max_features': ['auto','sqrt'],
    'n_estimators': [100, 200, 300, 1000]
}

recherche_hyperf=GridSearchCV(estimator=RandomForestRegressor(),param_grid=dico_paramrf,cv=5)
recherche_hyperf.fit(train_x, y_train)
print("les meilleurs parametres pour rf : {} avec\
            le score de {}".format(recherche_hyperf.best_params_,recherche_hyperf.best_score_))

# 8.2 : parametres  svr

dico_paramsvr = [{'C': [0.1,1, 5,8, 10,15,25],
              'kernel':['rbf','linear','poly','sigmoid'],
              'gamma': [0.0001, 0.0005, 0.001, 0.005]}]
recherche_hypersvr=GridSearchCV(estimator=SVR(),param_grid=dico_paramsvr,cv=5)
recherche_hypersvr.fit(train_x, y_train)
print("les meilleurs parametres pour svr : {} avec\
            le score de {}".format(recherche_hypersvr.best_params_,recherche_hypersvr.best_score_))

# 8.3 : parametres Adboost
      
dico_paramaba= {
 'n_estimators': [10,20,50, 100],
 'learning_rate' : [0.01, 0.05, 0.1, 0.5],
 'loss' : ['linear', 'square', 'exponential']
 }     
recherche_hyperaba=GridSearchCV(estimator=AdaBoostRegressor(),param_grid=dico_paramaba,cv=5)
recherche_hyperaba.fit(train_x, y_train)
print("les meilleurs parametres pour adaboost : {} avec\
            le score de {}".format(recherche_hyperaba.best_params_,recherche_hyperaba.best_score_))

# 8.4 : parametres XGBoost

dico_paramxgb={"max_depth":[3,5,7,10],"n_estimators":[10,20,50,100]}
recherche_hyperxgb=GridSearchCV(estimator=xgb.XGBRegressor(),param_grid=dico_paramxgb,cv=5)
recherche_hyperxgb.fit(train_x, y_train)
print("les meilleurs parametres pour XGBoost : {} avec\
            le score de {}".format(recherche_hyperxgb.best_params_,recherche_hyperxgb.best_score_))


# 9 : Apprentissage des methodes avec des parametres optimisés

modelxgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=10,max_depth=5)
modelxgb.fit(train_x, y_train,
        eval_set=[(train_x, y_train), (test_x, y_test)],
        early_stopping_rounds=100,
       verbose=0)    
"""
les meilleurs parametres pour XGBoost : {'max_depth': 5, 
'n_estimators': 10} avec le score de -5.723872878980228 """   

modelAbaB = AdaBoostRegressor(n_estimators=10,learning_rate=0.05,loss='square')
modelAbaB.fit(train_x, y_train)
"""
les meilleurs parametres pour adaboost : {'learning_rate': 0.05,
 'loss': 'square', 'n_estimators': 10} avec   le score de -7.784516217816194"""

modelRandomF=RandomForestRegressor(n_estimators=300,criterion='mse',max_depth=20,max_features='sqrt')
modelRandomF.fit(train_x, y_train)
"""les meilleurs parametres pour rf : {'bootstrap': True,
 'max_depth': 20, 'max_features': 'sqrt', 'n_estimators': 300} 
      le score de -6.033495048838132 """

modelSVR=SVR(kernel='sigmoid',gamma=0.0001,C=0.1)
modelSVR.fit(train_x, y_train)
"""
les meilleurs parametres pour svr : {'C': 0.1, 
 'gamma': 0.0001, 'kernel': 'sigmoid'} avec
 le score de -3.2885187425585927
 """
 

# 10 : Calcul des performances

# 10.1 : Calcul du r2_score
    
predTestXGB= modelxgb.predict(test_x)
predTestAdab=modelAbaB.predict(test_x)
predTestRandf=modelRandomF.predict(test_x)
predTestSvr=modelSVR.predict(test_x)


print(" le r2_score du modèle xgb Test est :",r2_score(y_test,predTestXGB))                                       
print("le r2_score du modèle Adaboost Test ",r2_score(y_test,predTestAdab))
print("le r2_score du modèle Random Forest Test ", r2_score(y_test,predTestRandf))
print("le r2_score du modèle SVR Test ",r2_score(y_test,predTestSvr)) 


# 10.2 : - Calcul des mean squarre avec paramettres optimisés



print(" modèle xgb test : ",mean_squared_error(y_test,predTestXGB))                                       
print(" modèle Adaboost test: ",mean_squared_error(y_test,predTestAdab))
print("modèle Random Forest test: ", mean_squared_error(y_test,predTestRandf))
print("modèle SVR test: ",mean_squared_error(y_test,predTestSvr)) 

# 11 - Visualisation des données en fonction de chaque modèle


data_total['cpu Prediction XGBoost']=modelxgb.predict(x_total)
data_total['cpu Prediction Adaboost']=modelAbaB.predict(x_total)
data_total['cpu Prediction Random F']=modelRandomF.predict(x_total)
data_total['cpu Prediction SVR']=modelSVR.predict(x_total)

train['cpu Prediction XGBoost']=modelxgb.predict(train_x)
train['cpu Prediction Adaboost']=modelAbaB.predict(train_x)
train['cpu Prediction Random F']=modelRandomF.predict(train_x)
train['cpu Prediction SVR']=modelSVR.predict(train_x)

print(" modèle xgb test : ",mean_squared_error(y_total,modelxgb.predict(x_total)))                                       
print(" modèle Adaboost test: ",mean_squared_error(y_total,modelAbaB.predict(x_total)))
print("modèle Random Forest test: ", mean_squared_error(y_total,modelRandomF.predict(x_total)))
print("modèle SVR test: ",mean_squared_error(y_total,modelSVR.predict(x_total))) 


# 11.1 - Visualisation des données en fonction de chaque modèle

_=train[['cpu','cpu Prediction XGBoost','cpu Prediction Adaboost','cpu Prediction Random F']].plot(figsize=(15, 5))
  
_ =data_total[['cpu','cpu Prediction XGBoost','cpu Prediction Adaboost','cpu Prediction Random F']].plot(figsize=(15, 5))
 
# 11.2 - Visualisation des données d'entrainement et Total sur les deux meilleurs modèle

x_total, y_total
plt.figure(figsize=(18,18))
plt.subplot(2,2,1)
plt.plot(train['date'],train['cpu'],color = 'green', label = 'Données d''entrainement')
plt.plot(train['date'],train['cpu Prediction Random F'],"b:o", color = 'blue', label = 'Prédiction Random Forest')
plt.plot(train['date'],train['cpu Prediction XGBoost'],"r--",color = 'red', label = 'Prédiction modèle XGBoost')
plt.ylabel('% CPU')
plt.xlabel('Dates et heures')
plt.legend(loc = 'best')
plt.title('Prédiction modèle XGBoost et Random Forest sur les données d''entrainement(% CPU) ')
plt.subplot(2,2,2)
plt.plot(data_total['date'],data_total['cpu'],color = 'green', label = 'Ensemble des Données')
plt.plot(data_total['date'],data_total['cpu Prediction Random F'],"b:o", color = 'blue', label = 'Prédiction Random Forest')
plt.plot(data_total['date'],data_total['cpu Prediction XGBoost'],"r--",color = 'red', label = 'Prédiction modèle XGBoost')
plt.ylabel('% CPU')
plt.xlabel('Dates et heures')
plt.legend(loc = 'best')
plt.title('Prédiction modèle XGBoost et Random Forest sur l''ensemble des données (% CPU) ')
plt.show()





