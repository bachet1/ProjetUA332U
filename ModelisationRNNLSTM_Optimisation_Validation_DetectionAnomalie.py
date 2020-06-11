# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:46:05 2020

@author: luc
Modelisation par RNN - LSTM

"""

# 1 - : Importation des librairies and packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error,make_scorer
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from keras.utils import plot_model
from keras.models import load_model
import seaborn as sns

# 2 - : Importing the training set

dataset_training=pd.read_csv("https://raw.githubusercontent.com/oreilly-mlsec/book-resources/599669c7124dffb65ea7f6e0b7626df32496b1d6/chapter3/datasets/cpu-utilization/cpu-train-a.csv", sep=",")
dataset_test=pd.read_csv("https://raw.githubusercontent.com/oreilly-mlsec/book-resources/599669c7124dffb65ea7f6e0b7626df32496b1d6/chapter3/datasets/cpu-utilization/cpu-test-a.csv", sep=",")
dataset_total_full=pd.read_csv("https://raw.githubusercontent.com/oreilly-mlsec/book-resources/599669c7124dffb65ea7f6e0b7626df32496b1d6/chapter3/datasets/cpu-utilization/cpu-full-a.csv", sep=",")
dataset_validation=pd.read_csv("https://raw.githubusercontent.com/oreilly-mlsec/book-resources/599669c7124dffb65ea7f6e0b7626df32496b1d6/chapter3/datasets/cpu-utilization/cpu-train-b.csv", sep=",")

training_set = dataset_training['cpu'].values
test_set = dataset_test['cpu'].values
real_utilisation_cpu = dataset_total_full['cpu'].values
dataset_validation=dataset_validation['cpu'].values

# 3 - : Feature Scaling


sc = MinMaxScaler(feature_range = (0, 1))
training_set= training_set.reshape(-1,1)
test_set= test_set.reshape(-1,1)

training_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.fit_transform(test_set)

real_utilisation_cpu=real_utilisation_cpu.reshape(-1,1)
real_utilisation_cpu_scaled = sc.fit_transform(real_utilisation_cpu)

# 4 - : Préparation des données pour le  retard(lags) timesteps 

lags=5
def preparation_data(data,lags):
	x_train = []
	y_train = []
	for i in range(lags,len(data)):
		x_train.append(data[i-lags:i, 0])
		y_train.append(data[i, 0])
	return np.array(x_train), np.array(y_train)


x_train, y_train = preparation_data(training_set_scaled, lags)
x_test, y_test = preparation_data(test_set_scaled, lags)

x_real_utilisation, y_real_utilisation = preparation_data(real_utilisation_cpu_scaled, lags)


# 5 : - Building the RNN and LSTM

# 5.1 : -  Reshaping

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 5.2 : - Initialising the RNN

start=time()

regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(LSTM(units = 128, return_sequences = True))
regressor.add(LSTM(units = 64, return_sequences = True))
regressor.add(LSTM(units = 32, return_sequences = True))
regressor.add(LSTM(units = 64, return_sequences = True))
regressor.add(LSTM(units = 128, return_sequences = True))
regressor.add(LSTM(units = 100))
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error',
                  metrics=['accuracy'])

# 5.3 : - Visualisation du modèle 

print(regressor.summary())

# 5.4 : - Fitting the RNN to the Training set

history=regressor.fit(x_train, y_train, epochs = 300, batch_size = 1,
              validation_data=(x_test, y_test),verbose=1)


elapsed=time()-start

print('duree totale est de :',elapsed/60)

print(regressor.summary())

plot_model(regressor,show_shapes=True, to_file='regressor.png')


# 6 : - Making the predictions and visualising the results


# 6.1 : - Plot training & validation accuracy values

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.subplot(1,2,2)
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()  

# 6.2 : - Calcul des erreurs

predictiontest= regressor.predict(x_test)
predictiontest= sc.inverse_transform(predictiontest)
y_test=sc.inverse_transform(y_test.reshape(-1, 1))
print("MAE sur données test :", mean_absolute_error(y_test, predictiontest))
print("MSE sur données test:",mean_squared_error(y_test, predictiontest))
print("RMSE sur données test:", sqrt(mean_squared_error(y_test, predictiontest)))


# 6.3 : - Graphique de reajustement sur les données test


test_set_reajuste=test_set[lags:len(predictiontest)+lags,]

plt.plot(test_set_reajuste, color = 'green', label = 'Utilisation réelle du cpu sur les données test')
plt.plot(predictiontest, color = 'blue', label = 'Prediction du modèle sur les données test')
plt.title('Prédiction du modèle sur les données test')
plt.text(10,1.,'MSE=%2.4f'%(mean_squared_error(y_test, predictiontest)))
plt.xlabel('Temps')
plt.ylabel('% cpu')
plt.legend()
plt.show()

#plt.text(30,1.8,'MSE=%2.4f''%''%(mean_squared_error(y_test, predictiontest)))
 

# 6.4 : - Metrics de validation du modèle

# 6.4.1 :  Prédiction sur l'ensemble des données

x_real_utilisation_scaled = np.reshape(x_real_utilisation, (x_real_utilisation.shape[0], x_real_utilisation.shape[1], 1))
predicted_utilisation_cpu = regressor.predict(x_real_utilisation_scaled)
predicted_utilisation_cpu_transf = sc.inverse_transform(predicted_utilisation_cpu)

y_real_utilisation_scaled=sc.inverse_transform(y_real_utilisation.reshape(-1, 1))

# 6.4.2 : - Metrics de validation du modèle et visualisation

print("MAE sur l'ensemble des données:", mean_absolute_error(y_real_utilisation_scaled, predicted_utilisation_cpu_transf ))
print("MSE sur l'ensemble des données:", mean_squared_error(y_real_utilisation_scaled, predicted_utilisation_cpu_transf))
print("RMSE sur l'ensemble des données:", sqrt(mean_squared_error(y_real_utilisation_scaled, predicted_utilisation_cpu_transf)))


real_utilisation_cpu_reajuste=real_utilisation_cpu[lags:len(predicted_utilisation_cpu_transf)+lags,]


plt.plot(real_utilisation_cpu_reajuste, color = 'green', label = 'Ensemble des Données en % CPU')
plt.plot(predicted_utilisation_cpu_transf, color = 'blue', label = 'Prédiction du modèle RNN-LSTM')
plt.text(210,0.75,'MSE=%2.4f'%(mean_squared_error(y_real_utilisation_scaled, predicted_utilisation_cpu_transf)))
plt.title('Modèle RNN-LSTM(lags=5) versus ensemble des Données (%CPU)')
plt.xlabel('Temps')
plt.ylabel('% CPU')
plt.legend()
plt.show()



# 6.4.3 : - Prédiction sur les données de validation


dataset_validation=dataset_validation.reshape(-1,1)
dataset_validation_t = sc.fit_transform(dataset_validation)
x_validation_utilisation, y_validation_utilisation_scaled = preparation_data(dataset_validation_t, lags)

x_validation_utilisation_scaled = np.reshape(x_validation_utilisation, (x_validation_utilisation.shape[0], x_validation_utilisation.shape[1], 1))
predicted_x_validation_cpu = regressor.predict(x_validation_utilisation_scaled)
predicted_x_validation_cpu_transf = sc.inverse_transform(predicted_x_validation_cpu)

y_validation_utilisation_scaled=sc.inverse_transform(y_validation_utilisation_scaled.reshape(-1, 1))


print("MAE sur les données validation:", mean_absolute_error(y_validation_utilisation_scaled, predicted_x_validation_cpu_transf ))
print("MSE sur les données validation:", mean_squared_error(y_validation_utilisation_scaled, predicted_x_validation_cpu_transf))
print("RMSE sur les données validation:", sqrt(mean_squared_error(y_validation_utilisation_scaled, predicted_x_validation_cpu_transf)))


dataset_validation_reajuste=dataset_validation[lags:len(predicted_x_validation_cpu_transf)+lags,]
plt.plot(dataset_validation_reajuste, color = 'green', label = 'Données de validation % CPU')
plt.plot(predicted_x_validation_cpu_transf, color = 'blue', label = 'Prédiction du modèle RNN-LSTM')
plt.title('Modèle RNN-LSTM(lags=5) versus Données validation (%CPU)')
plt.text(210,0.55,'MSE=%2.4f'%(mean_squared_error(y_validation_utilisation_scaled, predicted_x_validation_cpu_transf)))
plt.xlabel('Temps')
plt.ylabel('% CPU')
plt.legend()
plt.show()

# 6.4.4 : -  Visualisation des graphiques deux à deux

plt.figure(figsize=(18,15))
plt.subplot(1,2,1)
plt.plot(real_utilisation_cpu_reajuste, color = 'green', label = 'Ensemble des Données en % CPU')
plt.plot(predicted_utilisation_cpu_transf, color = 'blue', label = 'Prédiction du modèle RNN-LSTM')
plt.text(210,0.55,'MSE=%2.4f'%(mean_squared_error(y_real_utilisation_scaled, predicted_utilisation_cpu_transf)))
plt.title('Ajustement du Modèle RNN-LSTM(lags=5) sur l''ensemble des Données (%CPU)')
plt.xlabel('Temps')
plt.ylabel('% CPU')
plt.legend()
plt.subplot(1,2,2)
plt.plot(dataset_validation_reajuste, color = 'green', label = 'Données de validation % CPU')
plt.plot(predicted_x_validation_cpu_transf, color = 'blue', label = 'Prédiction du modèle RNN-LSTM')
plt.title('Ajustement du Modèle RNN-LSTM(lags=5) sur les Données de validation (%CPU)')
plt.text(210,0.55,'MSE=%2.4f'%(mean_squared_error(y_validation_utilisation_scaled, predicted_x_validation_cpu_transf)))
plt.xlabel('Temps')
plt.ylabel('% CPU')
plt.legend()
plt.legend()
plt.show()

# 7 : Validation du modèle par validation croisée

def build_regressor():
    regressor = Sequential()
    regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    regressor.add(LSTM(units = 128, return_sequences = True))
    regressor.add(LSTM(units = 64, return_sequences = True))
    regressor.add(LSTM(units = 32, return_sequences = True))
    regressor.add(LSTM(units = 64, return_sequences = True))
    regressor.add(LSTM(units = 128, return_sequences = True))
    regressor.add(LSTM(units = 100))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])
    return regressor

regressoroptimiser=KerasRegressor(build_fn=build_regressor,batch_size=1,epochs=300)

precision=cross_val_score(estimator=regressoroptimiser,X=x_train,y=y_train,cv=10)
precision.mean()
precision.std()

print('precision:',precision)

"""
precision=[-0.00120841, -0.00632367, -0.00748501, -0.001921 ,  -0.00025073, -0.00838645,
 -0.00582629, -0.00436221, -0.00793974, -0.02732443]

"""
plt.plot(precision, color = 'blue')
plt.text(2,-0.015,'$\mu=%.4f,\\sigma=%.4f$'%(precision.mean(),precision.std()))
plt.title('RNN-LSTM(lags=5) mean_squared_error par validation croisée')
plt.show()


# 8 : Recherche des meilleurs paramettres du modèle

start=time()

# 8.1 : Recherche du meilleurs du  modèle (optimizer, batch_size)

def build_regressor(optimizer):
    regressor = Sequential()
    regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    regressor.add(LSTM(units = 128, return_sequences = True))
    regressor.add(LSTM(units = 64, return_sequences = True))
    regressor.add(LSTM(units = 32, return_sequences = True))
    regressor.add(LSTM(units = 64, return_sequences = True))
    regressor.add(LSTM(units = 128, return_sequences = True))
    regressor.add(LSTM(units = 100))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = optimizer, loss = 'mean_squared_error',metrics=['accuracy'])
    return regressor

model=KerasRegressor(build_fn=build_regressor,verbose=1)
parameters={"optimizer":['adam','rmsprop'],
            "epochs":[55,100],
            "batch_size":[1,2,3,5]
            }


grid_search=GridSearchCV(estimator=model , param_grid=parameters, scoring=make_scorer(mean_squared_error),cv=5)

grid_search=grid_search.fit(x_train,y_train)

best_parameters=grid_search.best_params_
best_score=grid_search.best_score_

elapsed=time()-start

print(" meilleurs score:",best_score)
print(" meilleurs parametres:",best_parameters)
print(" Durée Totale:",elapsed/60)

"""
 meilleurs score: 0.00880390799525373
 meilleurs parametres: {'batch_size': 5, 'epochs': 100, 'optimizer': 'rmsprop'}
 Durée Totale: 52325.95503425598 secondes soient 872.099250570933 minutes soient 14.53449 heures
 
"""
# 8.2 : Recherche du meilleurs Epoch du  modèle

start=time()

def build_regressor():
    regressor = Sequential()
    regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    regressor.add(LSTM(units = 128, return_sequences = True))
    regressor.add(LSTM(units = 64, return_sequences = True))
    regressor.add(LSTM(units = 32, return_sequences = True))
    regressor.add(LSTM(units = 64, return_sequences = True))
    regressor.add(LSTM(units = 128, return_sequences = True))
    regressor.add(LSTM(units = 100))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',metrics=['accuracy'])
    return regressor

model=KerasRegressor(build_fn=build_regressor,batch_size=5,verbose=1)
parameters={"epochs":[100,150,200,300,500,600]}

grid_search=GridSearchCV(estimator=model , param_grid=parameters, scoring=make_scorer(mean_squared_error),cv=2)

grid_search=grid_search.fit(x_train,y_train)

best_parameters=grid_search.best_params_
best_score=grid_search.best_score_

elapsed=time()-start

print(" meilleurs score:",best_score)
print(" meilleurs parametres:",best_parameters)
print(" Durée Totale(Minutes):",elapsed/60)

"""
 meilleurs score: 0.009108856431690971; meilleurs parametres: {'epochs': 500}
 Durée Totale(Minutes): 81.92450955708821"""

# 9 : Entrainement du modele avec les parametres optimisés

start=time()

# Fonction pour le modèle

def build_model():
    start=time()
    print("Debut d'entrainement du modèle")
    model = Sequential()
    model.add(LSTM(units = 100, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(LSTM(units = 128, return_sequences = True))
    model.add(LSTM(units = 64, return_sequences = True))
    model.add(LSTM(units = 32, return_sequences = True))
    model.add(LSTM(units = 64, return_sequences = True))
    model.add(LSTM(units = 128, return_sequences = True))
    model.add(LSTM(units = 100))
    model.add(Dense(units = 1))
    model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',metrics=['accuracy'])
    history=model.fit(x_train, y_train, epochs = 500, batch_size = 5,
              validation_data=(x_test, y_test),verbose=1)
    elapsed=time()-start
    print("L'entranement du modèle a une durée(Minutes) :",elapsed/60)
    return  model,history

# Entrainement du modèle

model=build_model() #L'entranement du modèle a une durée(Minutes) 16.67389272848765

start=time()
model = Sequential()
model.add(LSTM(units = 100, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(units = 128, return_sequences = True))
model.add(LSTM(units = 64, return_sequences = True))
model.add(LSTM(units = 32, return_sequences = True))
model.add(LSTM(units = 64, return_sequences = True))
model.add(LSTM(units = 128, return_sequences = True))
model.add(LSTM(units = 100))
model.add(Dense(units = 1))
model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
              metrics=['accuracy'])
print(model.summary())

history=model.fit(x_train, y_train, epochs = 500, batch_size = 5,
                  validation_data=(x_test, y_test),verbose=1)

elapsed=time()-start
print("L'entranement du modèle a une durée(Minutes) :",elapsed/60)
"""
- val_accuracy: 0.0182 ; L'entranement du modèle a une durée(Minutes) : 17.206155995527904
"""
# 9.1 : - Visualisation du modèle 


plot_model(model.summary(),show_shapes=True, to_file='modelRNNlSTM.png')

# 9.2 : Plot training & validation accuracy values

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.subplot(1,2,2)
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()  


# 9.3 : Evaluation des metrics du meilleurs du  modèle

# 9.3.1 : Calcul du score

scores = model.evaluate(x_test, y_test,verbose=0)

print("une de:",scores) # [0.007295613242736594, 0.0181818176060915]

# Le premier élément de scores renvoie la fonction de coût sur la base de test, 
# le second élément renvoie le taux de bonne détection (accuracy).
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100)) # loss: 0.73%
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)) # accuracy: 1.82%

# 9.3.2 : Calcul du mean squared error et autres mesures associées

# metrics sur les données Test

predictiontest= model.predict(x_test)
predictiontest= sc.inverse_transform(predictiontest)
y_test=sc.inverse_transform(y_test.reshape(-1, 1))
print("MAE sur données test :", mean_absolute_error(y_test, predictiontest))
print("MSE sur données test:",mean_squared_error(y_test, predictiontest))
print("RMSE sur données test:", sqrt(mean_squared_error(y_test, predictiontest)))

# metrics sur les données validations

predicted_x_validation_cpu = model.predict(x_validation_utilisation_scaled)
predicted_x_validation_cpu_transf = sc.inverse_transform(predicted_x_validation_cpu)
y_validation_utilisation_scaled=sc.inverse_transform(y_validation_utilisation_scaled.reshape(-1, 1))
print("MAE sur les données validation:", mean_absolute_error(y_validation_utilisation_scaled, predicted_x_validation_cpu_transf ))
print("MSE sur les données validation:", mean_squared_error(y_validation_utilisation_scaled, predicted_x_validation_cpu_transf))
print("RMSE sur les données validation:", sqrt(mean_squared_error(y_validation_utilisation_scaled, predicted_x_validation_cpu_transf)))


test_set_reajuste=test_set[lags:len(predictiontest)+lags,]

plt.figure(figsize=(18,15))
plt.subplot(2,2,1)
plt.plot(test_set_reajuste, color = 'green', label = 'Utilisation réelle du cpu sur les données test')
plt.plot(predictiontest, color = 'blue', label = 'Prediction du modèle sur les données test')
plt.title('Prédiction du modèle sur les données test')
plt.text(10,1.,'MSE=%2.4f'%(mean_squared_error(y_test, predictiontest)))
plt.xlabel('Temps')
plt.ylabel('% cpu')
plt.legend()

plt.figure(figsize=(18,15))
plt.subplot(2,2,1)
plt.plot(real_utilisation_cpu_reajuste, color = 'green', label = 'Ensemble des Données en % CPU')
plt.plot(predicted_utilisation_cpu_transf, color = 'blue', label = 'Prédiction du modèle RNN-LSTM')
plt.text(210,0.55,'MSE=%2.4f'%(mean_squared_error(y_real_utilisation_scaled, predicted_utilisation_cpu_transf)))
plt.title('Ajustement du Modèle RNN-LSTM(lags=5) sur l''ensemble des Données (%CPU)')
plt.xlabel('Temps')
plt.ylabel('% CPU')
plt.legend()
plt.subplot(2,2,2)
plt.plot(dataset_validation_reajuste, color = 'green', label = 'Données de validation % CPU')
plt.plot(predicted_x_validation_cpu_transf, color = 'blue', label = 'Prédiction du modèle RNN-LSTM')
plt.title('Ajustement du Modèle RNN-LSTM(lags=5) sur les Données de validation (%CPU)')
plt.text(210,0.55,'MSE=%2.4f'%(mean_squared_error(y_validation_utilisation_scaled, predicted_x_validation_cpu_transf)))
plt.xlabel('Temps')
plt.ylabel('% CPU')
plt.legend()
plt.legend()
plt.show()


# 10 : Enregistrement du modèle

# creates a HDF5 file 'my_model.h5'

model.save('modelRnnLSTMProjetSecurite.h5') 

#creates a HDF5 file 'my_model.h5' et model = load_model('modelRnnLSTMProjetSecurite.h5')
# pour le detruire del model  # deletes the existing model

# 11 - Detection d'anomalie avec les RNN - LSTM


predicted_utilisation_cpu = model.predict(x_real_utilisation_scaled)
predicted_utilisation_cpu_transf = sc.inverse_transform(predicted_utilisation_cpu)

# y_real_utilisation_scaled=sc.inverse_transform(y_real_utilisation.reshape(-1, 1))

# 11.1 - Calcul des erreurs de prédiction

erreurs=real_utilisation_cpu_reajuste-predicted_utilisation_cpu_transf

# 11.2 - Graphique de Distribution des erreurs de prédiction

plt.figure(figsize=(18,15)) 
sns.distplot(erreurs,bins=50,kde=True)
plt.title('Distribution des erreurs sur l''ensemble des Données')

# On observe deux valeurs inf a -0.2 et sup 0.2

THRESHOLD_Inf=-0.2
THRESHOLD_Sup=0.2
THRESHOLD=0.2
         
# 11.3  : Constitution des anomalies et du niveau de seuil de detection 

data_score_df['date']=pd.to_datetime(dataset_total_full[lags:].datetime)
data_score_df['loss']=pd.DataFrame(erreurs)
data_score_df['cpu']=dataset_total_full[lags:].cpu
data_score_df['cpu_predit']=predicted_utilisation_cpu_transf

data_score_df['THRESHOLD']=THRESHOLD
data_score_df['THRESHOLD_Inf']=THRESHOLD_Inf
data_score_df['THRESHOLD_Sup']=THRESHOLD_Sup


# 11.3.1 : Visualisation du niveau de la courbe des pertes et du seuil

plt.figure(figsize=(18,15)) 
plt.subplot(2,2,1)
sns.distplot(erreurs,bins=50,kde=True)
plt.title('Distribution des erreurs de prédiction sur l"ensemble des Données')
plt.subplot(2,2,2)
plt.plot(data_score_df['date'],data_score_df['loss'],label='Résidus(loss)')
plt.plot(data_score_df['date'],data_score_df['THRESHOLD_Sup'],label='seuil inferieur')
plt.plot(data_score_df['date'],data_score_df['THRESHOLD_Inf'],label='seuill superieur')
plt.xticks(rotation=25)
plt.title('Distribution des erreurs de prédiction et Niveau du seuil')
plt.legend()

# 11.3.2 : Tableau des anomalies


data_score_df.loc[data_score_df['loss']<=THRESHOLD_Inf,'anomalie']=True
data_score_df.loc[data_score_df['loss']>=THRESHOLD_Sup,'anomalie']=True


# 11.3.3 : : Visualisation des anomalies sur le graphique 

anomalies=data_score_df[data_score_df.anomalie==True]

print('nombre d''anomalies: ',anomalies.shape[0])
plt.figure(figsize=(18,15))
plt.plot(data_score_df.date,data_score_df.cpu,'blue',label='cpu d''origine');
plt.plot(data_score_df.date,data_score_df.cpu_predit,marker='v',label='prediction data test prédite');
sns.scatterplot(
    anomalies.date,anomalies.cpu,
    color=sns.color_palette()[3],
    s=300,
    label='anomalie')
plt.xticks(rotation=25)
plt.xlabel('Minutes,heures, jour') 
plt.ylabel('% cpu') 
plt.title('Le nombre d''anomalies au seuil erreur(- 0.3 et 0.3) données Totale: %2.0f'%(anomalies.shape[0]))
plt.legend(loc='upper left');
plt.show()

#12 : Fonction de creation d'anomalie
# Management des anomalies en fonction du seuil
def anomalies(data,data_reel_reajuste,data_predict,seuil1,seuil2):
    erreurs=data_reel_reajuste-data_predict
    THRESHOLD_Inf,THRESHOLD_Sup =seuil1,seuil2
    data_score_df['date']=pd.to_datetime(data[lags:].datetime)
    data_score_df['loss']=pd.DataFrame(erreurs)
    data_score_df['cpu']=dataset_total_full[lags:].cpu
    data_score_df['cpu_predit']=predicted_utilisation_cpu_transf
    data_score_df.loc[data_score_df['loss']<=THRESHOLD_Inf,'anomalie']=True
    data_score_df.loc[data_score_df['loss']>=THRESHOLD_Sup,'anomalie']=True
    anomalies=data_score_df[data_score_df.anomalie==True]
    plt.figure(figsize=(18,15))
    plt.subplot(2,2,1)
    sns.distplot(erreurs,bins=50,kde=True)
    plt.title('Distribution des erreurs de prédiction sur l"ensemble des Données')
    plt.subplot(2,2,2)
    plt.plot(data_score_df['date'],data_score_df['loss'],label='Résidus(loss)')
    plt.plot(data_score_df['date'],data_score_df['THRESHOLD_Sup'],label='seuil inferieur')
    plt.plot(data_score_df['date'],data_score_df['THRESHOLD_Inf'],label='seuill superieur')
    plt.xticks(rotation=25)
    plt.title('Distribution des erreurs de prédiction et Niveau du seuil')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(data_score_df.date,data_score_df.cpu,'blue',label='cpu d''origine')
    plt.plot(data_score_df.date,data_score_df.cpu_predit,marker='v',label='prediction data test prédite')
    sns.scatterplot(anomalies.date,anomalies.cpu,color=sns.color_palette()[3], s=300,label='anomalie')
    plt.xticks(rotation=25)
    plt.xlabel('Minutes,heures, jour')
    plt.ylabel('% cpu')
    plt.title('Le nombre d''anomalies au seuil erreur(- 0.2 et 0.2) données Totale: %2.0f'%(anomalies.shape[0]))
    plt.legend(loc='upper left')
    plt.show()
# Management des anomalies en fonction du seuil    
anomalies(dataset_total_full,real_utilisation_cpu_reajuste,predicted_utilisation_cpu_transf,-0.2,0.2)
anomalies(dataset_validation,dataset_validation_reajuste,predicted_x_validation_cpu_transf,-0.2,0.2)
anomalies(dataset_test,test_set_reajuste,predictiontest,-0.2,0.2)
