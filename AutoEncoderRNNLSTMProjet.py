# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:03:39 2020

@author: luc

"""

# 1 -  importation des librairies

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# 2 - importation des données 

dataset_full=pd.read_csv("https://raw.githubusercontent.com/oreilly-mlsec/book-resources/599669c7124dffb65ea7f6e0b7626df32496b1d6/chapter3/datasets/cpu-utilization/cpu-full-a.csv",sep=',')
dataset_training=pd.read_csv("https://raw.githubusercontent.com/oreilly-mlsec/book-resources/599669c7124dffb65ea7f6e0b7626df32496b1d6/chapter3/datasets/cpu-utilization/cpu-train-a.csv", sep=",")
dataset_test=pd.read_csv("https://raw.githubusercontent.com/oreilly-mlsec/book-resources/599669c7124dffb65ea7f6e0b7626df32496b1d6/chapter3/datasets/cpu-utilization/cpu-test-a.csv", sep=",")

dataset_full['date']=pd.to_datetime(dataset_full.datetime)
dataset_training['date']=pd.to_datetime(dataset_training.datetime)
dataset_test['date']=pd.to_datetime(dataset_test.datetime)
dataset_full=dataset_full[['date','cpu']]

# 3 - Séparation des données d'apprentissage et de test

# 3.1 - : Processing des données

scaler = MinMaxScaler(feature_range = (0, 1))

train=scaler.fit_transform(dataset_training[['cpu']])

test=scaler.fit_transform(dataset_test[['cpu']])

dataset=scaler.fit_transform(dataset_full[['cpu']])

#3.2 - Verification

print(train.head(),test.head())

#date,close

timesteps = 5
n_features=1

# 3.3 : fonction de préparation des données

def preparation_data(data,lags):
	x_train = []
	y_train = []
	for i in range(lags,len(data)):
		x_train.append(data[i-lags:i, 0])
		y_train.append(data[i, 0])
	return np.array(x_train), np.array(y_train)

# 3.4 : reshaping des données pour etre mis en RNN-LSTM For [sample,timesteps,n_features]
x_train, y_train=preparation_data(train,timesteps)
x_test, y_test=preparation_data(test,timesteps)
x_dataset_full, y_dataset_full=preparation_data(dataset,timesteps)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_dataset_full = np.reshape(x_dataset_full, (x_dataset_full.shape[0], x_dataset_full.shape[1], 1))

print(x_train.shape,x_test.shape,x_dataset_full.shape)
# (415, 5, 1) (55, 5, 1) (475, 5, 1)

# 4 : création et mis en place du RNN-LSTM auto-Encoder

start=time()
# define model
model = Sequential()
model.add(LSTM(100, input_shape=(timesteps,n_features), return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32,return_sequences=False))
model.add(RepeatVector(timesteps))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100,return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.summary()
# fit model
history=model.fit(x_dataset_full, x_dataset_full, epochs=500,batch_size=5,verbose=1,
                  validation_split=0.10)

# 4.1.4: Architecture

plot_model(model, show_shapes=True, to_file='Architecture hybride Auto-Encoder-Prédicteur.png')
elapsed=time()-start
print('duree totale est de :',elapsed/60)
model.save('modelAutoEncoderProjet.h5') 
# model = load_model('modelAutoEncoderProjet.h5')
# 4.2 : Graphique loss & val_loss
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.title('Loss & Epoch :Data training & Data Test')
plt.legend()
# 4.2 : Graphique loss & val_loss
plt.figure(figsize=(18,15)) 
plt.subplot(2,2,1)
plt.plot(history.history['loss'],label='Données d"entrainement')
plt.plot(history.history['val_loss'],label='Données Test')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.title('Loss & Epoch :Données d"entrainement et Données Test')
plt.legend()

# 4.3 :  Prédiction des données Test et d'entrainement par le modèle hybride
# model = load_model('modelAutoEncoderProjet.h5')
pred_x_train = model.predict(x_train)
pred_x_test = model.predict(x_test, verbose=0)
pred_x_dataset_full = model.predict(x_dataset_full, verbose=0)

# 4.3.2 : reshapping ou flattering des données par une fonction

def flatten(x):
    flattened_x=np.empty((x.shape[0],x.shape[2]))
    for i in range(x.shape[0]):
        flattened_x[i]=x[i,(x.shape[1]-1),:]
    return(flattened_x)


# 4.3.3 : Visualisation des données origines et des données reconstituées

# plt.plot(test.date,test.cpu,'r--',label='data test')

dataset_test_set=dataset_test[timesteps:]

plt.subplot(2,2,2)

# plt.plot(dataset_test_set.date,dataset_test_set.cpu,marker='v',label='Données Test')

plt.plot(scaler.inverse_transform(flatten(x_test)),marker='v',label='Données Test')
plt.plot(scaler.inverse_transform(flatten(pred_x_test)),'r--',label='Données Test reconstituées')
plt.title('Ajustement données Test Origines & Réconstituées')
plt.legend()
plt.show()    

# 4.3.4 : calcul du mse 

train_mse_loss=np.mean(np.power(flatten(x_train)-flatten(pred_x_train),2),axis=1) 
test_mse_loss=np.mean(np.power(flatten(x_test)-flatten(pred_x_test ),2),axis=1) 
full_mse_loss=np.mean(np.power(flatten(x_dataset_full)-flatten(pred_x_dataset_full),2),axis=1) 

# 4.3.5 :calcul du seuil pour determiner les anomalies

Erreurs=flatten(x_dataset_full)-flatten(pred_x_dataset_full)
sns.distplot(Erreurs,bins=50,kde=True)
plt.title('Distribution des Erreurs dataset total')
plt.legend()

THRESHOLD=0.004
plt.figure(figsize=(18,15)) 
plt.subplot(2,2,1) 
sns.distplot(test_mse_loss,bins=50,kde=True)
plt.title('Distribution du MSE(mean Squared Error) Test')
plt.legend()

# 4.3.5 : Constitution de la data anomalies ; on joue alors sur le niveau du seuil ex:THRESHOLD=0.96
test_score_df = pd.DataFrame(index=dataset_test[timesteps:].date)
test_score_df['loss']=test_mse_loss
test_score_df['cpu']=scaler.inverse_transform(flatten(x_test))
test_score_df['cpu_predit']=scaler.inverse_transform(flatten(pred_x_test))
test_score_df['THRESHOLD']=THRESHOLD
test_score_df['anomalie']=test_score_df.loss>test_score_df.THRESHOLD

# 5 : graphique des anomalies

# 5.1 : graphique des anomalies, du niveau de la courbe des pertes et du seuil
plt.subplot(2,2,2) 
plt.plot(test_score_df.index,test_score_df.loss,label='loss')
plt.plot(test_score_df.index,test_score_df.THRESHOLD,label='threshold')
plt.xticks(rotation=25)
plt.title('Evolution MSE(mean Squared Error) données Test')
plt.legend()

# 5.2 : calcul des anomalies

anomalies= test_score_df[test_score_df.anomalie]

# 5.3 : affichage des anomalies sur le graphique

anomalies=test_score_df[test_score_df.anomalie==True]

print('nombre d''anomalies: ',anomalies.shape[0])

# plt.figure(figsize=(18,15))
plt.subplot(2,2,3)
plt.plot(test_score_df.index,test_score_df.cpu,'r+:',label='cpu d''origine');
plt.plot(test_score_df.index,test_score_df.cpu_predit,marker='v',label='prediction data test prédite');
sns.scatterplot(
    anomalies.index,anomalies.cpu,
    color=sns.color_palette()[3],
    s=52,
    label='anomalie')
plt.xticks(rotation=25)
plt.xlabel('Minutes,heures, jour')
plt.ylabel('% cpu')
plt.title('Niveau des anomalies - seuil MSE(0.004)- Données Test: %2.0f'%(anomalies.shape[0]) + ' anomalies')
plt.legend();

# ajouter nbre d'anomalie dans le graphique: print('nombre d''anomalies: ',anomalies.shape[0])
# A partir d'ici juste les essaies avec  les données de validation

# 6 - données validation

full_score_df=pd.DataFrame(index=dataset_full[timesteps:].date)
full_score_df['loss']=full_mse_loss
full_score_df['cpu']=scaler.inverse_transform(flatten(x_dataset_full))
full_score_df['cpu_predit']=scaler.inverse_transform(flatten(pred_x_dataset_full))
full_score_df['THRESHOLD']=THRESHOLD
full_score_df['anomalie']=full_score_df.loss>full_score_df.THRESHOLD

anomalies_full= full_score_df[full_score_df.anomalie]
anomalies_full=full_score_df[full_score_df.anomalie==True]
# plt.figure(figsize=(18,15))
plt.subplot(2,2,4)
plt.plot(full_score_df.index,full_score_df.cpu,'r+:',label='cpu d''origine');
plt.plot(full_score_df.index,full_score_df.cpu_predit,marker='v',label='prediction data validation');
sns.scatterplot(
    anomalies_full.index,anomalies_full.cpu,
    color=sns.color_palette()[3],
    s=52,
    label='anomalie')
plt.xticks(rotation=25)
plt.xlabel('Minutes,heures, jour') 
plt.ylabel('% cpu') 
plt.title('Niveau des anomalies - seuil MSE(0.004)- Données Totale : %2.0f'%(anomalies_full.shape[0]) + ' anomalies')
plt.legend()
plt.show();

#print('nombre d''anomalies val: ',anomalies_val.shape[0])

# definition d'une fonction

def anomalie(seuil,data,data_predict,erreur,label1,label2):
    THRESHOLD=seuil
    test_score_df=pd.DataFrame(index=data[timesteps:].date)
    test_score_df['loss']=erreur
    test_score_df['cpu']=scaler.inverse_transform(flatten(data))# x_test
    test_score_df['cpu_predit']=scaler.inverse_transform(flatten(data_predict)) #x_prediction_test
    test_score_df['THRESHOLD']=THRESHOLD
    test_score_df['anomalie']=test_score_df.loss>test_score_df.THRESHOLD
    plt.plot(test_score_df.index,test_score_df.loss,label='loss')# 5 : graphique des anomalies
    plt.plot(test_score_df.index,test_score_df.THRESHOLD,label='threshold')
    plt.xticks(rotation=25)
    plt.title(label1)
    plt.legend()
    anomalies= test_score_df[test_score_df.anomalie]
    anomalies=test_score_df[test_score_df.anomalie==True]
    plt.figure(figsize=(18,15))
    plt.subplot(2,1,1)
    plt.plot(test_score_df.index,test_score_df.cpu,'r+:',label='cpu d''origine');
    plt.plot(test_score_df.index,test_score_df.cpu_predit,marker='v',label='prediction data test prédite');
    sns.scatterplot(
    anomalies.index,anomalies.cpu,
    color=sns.color_palette()[3],
    s=52,
    label='anomalie')
    plt.xticks(rotation=25)
    plt.xlabel('Minutes,heures, jour')
    plt.ylabel('% cpu')
    plt.title('Niveau des anomalies - seuil MSE(0.001)- data validation avec : %2.0f'%(anomalies.shape[0]) + ' anomalies')
    plt.legend();
    print('nombre d''anomalies: ',anomalies.shape[0])

# 6 - calcul des anomalies par les erreures de prediction
THRESHOLD=0.06
plt.figure(figsize=(18,15)) 
plt.subplot(2,2,1) 
Erreurs=flatten(x_dataset_full)-flatten(pred_x_dataset_full)
sns.distplot(Erreurs,bins=50,kde=True)
plt.title('Distribution des Erreurs sur l"Ensemble des Données')
plt.legend()
# 4.3.5 : Constitution de la data anomalies ; on joue alors sur le niveau du seuil ex:THRESHOLD=0.96
test_score_df = pd.DataFrame(index=dataset_test[timesteps:].date)
test_score_df['loss']=flatten(x_test)-flatten(pred_x_test)
test_score_df['cpu']=scaler.inverse_transform(flatten(x_test))
test_score_df['cpu_predit']=scaler.inverse_transform(flatten(pred_x_test))
test_score_df['THRESHOLD']=THRESHOLD
test_score_df['anomalie']=test_score_df.loss>test_score_df.THRESHOLD

# 5 : graphique des anomalies

# 5.1 : graphique des anomalies, du niveau de la courbe des pertes et du seuil
plt.subplot(2,2,2) 
plt.plot(test_score_df.index,test_score_df.loss,label='loss')
plt.plot(test_score_df.index,test_score_df.THRESHOLD,label='threshold')
plt.xticks(rotation=25)
plt.title('Erreurs de reconstruction sur les données Test')
plt.legend()

# 5.2 : calcul des anomalies

anomalies= test_score_df[test_score_df.anomalie]

# 5.3 : affichage des anomalies sur le graphique

anomalies=test_score_df[test_score_df.anomalie==True]

print('nombre d''anomalies: ',anomalies.shape[0])

# plt.figure(figsize=(18,15))
plt.subplot(2,2,3)
plt.plot(test_score_df.index,test_score_df.cpu,'r+:',label='cpu d''origine');
plt.plot(test_score_df.index,test_score_df.cpu_predit,marker='v',label='prediction data test prédite');
sns.scatterplot(
    anomalies.index,anomalies.cpu,
    color=sns.color_palette()[3],
    s=52,
    label='anomalie')
plt.xticks(rotation=25)
plt.xlabel('Minutes,heures, jour')
plt.ylabel('% cpu')
plt.title('Nombre d"anomalies detectées au seuil Erreurs(0.06) sur les Données Test: %2.0f'%(anomalies.shape[0]))
plt.legend();

# ajouter nbre d'anomalie dans le graphique: print('nombre d''anomalies: ',anomalies.shape[0])

# 6 - données totale

full_score_df=pd.DataFrame(index=dataset_full[timesteps:].date)
full_score_df['loss']=flatten(x_dataset_full)-flatten(pred_x_dataset_full)
full_score_df['cpu']=scaler.inverse_transform(flatten(x_dataset_full))
full_score_df['cpu_predit']=scaler.inverse_transform(flatten(pred_x_dataset_full))
full_score_df['THRESHOLD']=THRESHOLD
full_score_df['anomalie']=full_score_df.loss>full_score_df.THRESHOLD

anomalies_full= full_score_df[full_score_df.anomalie]
anomalies_full=full_score_df[full_score_df.anomalie==True]
# plt.figure(figsize=(18,15))
plt.subplot(2,2,4)
plt.plot(full_score_df.index,full_score_df.cpu,'r+:',label='cpu d''origine');
plt.plot(full_score_df.index,full_score_df.cpu_predit,marker='v',label='prediction data validation');
sns.scatterplot(
    anomalies_full.index,anomalies_full.cpu,
    color=sns.color_palette()[3],
    s=52,
    label='anomalie')
plt.xticks(rotation=25)
plt.xlabel('Minutes,heures, jour') 
plt.ylabel('% cpu') 
plt.title('Nombre d"anomalies detectées au seuil Erreurs(0.06) sur les Données Totale : %2.0f'%(anomalies_full.shape[0]))
plt.legend()
plt.show();