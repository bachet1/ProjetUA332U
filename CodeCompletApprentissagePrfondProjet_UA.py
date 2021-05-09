# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:54:02 2020

@author: luc
"""

# I - : Importation des librairies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from keras.models import Sequential, Model,load_model
from keras.layers import Dropout,Dense
from keras import Input,layers
from keras.layers import LSTM, GRU,Bidirectional
from keras.layers import Conv1D,MaxPooling1D,GlobalMaxPooling1D
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score,make_scorer
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.utils import plot_model
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,StackingRegressor,VotingRegressor,GradientBoostingRegressor
#from sklearn.externals import joblib
import joblib
from joblib import dump, load
from datetime import datetime
from tensorflow.keras.models import Model, load_model,Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D,MaxPooling1D,LSTM, GRU,Bidirectional,GlobalMaxPooling1D
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score,make_scorer
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.python.keras.optimizers import TFOptimizer
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn.model_selection import GridSearchCV
from keras.utils import plot_model
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,StackingRegressor,VotingRegressor,GradientBoostingRegressor

# II - : Importing des données

dataset=pd.read_csv("https://raw.githubusercontent.com/oreilly-mlsec/book-resources/599669c7124dffb65ea7f6e0b7626df32496b1d6/chapter3/datasets/cpu-utilization/cpu-full-b.csv", sep=",")

data_total = pd.read_csv("https://raw.githubusercontent.com/oreilly-mlsec/book-resources/599669c7124dffb65ea7f6e0b7626df32496b1d6/chapter3/datasets/cpu-utilization/cpu-full-b.csv",parse_dates=[0], index_col=0)

dataset=data_total

# III -  Separation des données : Entrainement et de Test

# III.1 - : Constitution des échantillons

train_size=int(len(dataset)*.85)
test_size=int(len(dataset)*.15)
x_trainning,x_testing=dataset.iloc[0:train_size],dataset.iloc[(train_size+1):(train_size+test_size)]


# III.2 - : Normalisation des Données

x_train_data=x_trainning['cpu'].values
x_test_data=x_testing['cpu'].values

sc = MinMaxScaler(feature_range = (0, 1))
# data1= x_train_data.reshape(-1,1)
data_train = sc.fit_transform(x_train_data.reshape(-1,1))
data_test = sc.fit_transform(x_test_data.reshape(-1,1))


# III.3 - : Préparation des données pour les algorithmes d'apprentissages(retard(lags), timesteps) 

"""' 
Nous allons utiliser deux fonctions:la premiere effectue directement la preparation des données. 
La seconde , la preparation et la normalisation des données 

"""

def preparation_data(data,lags):
        x_train = []
        y_train = []
        for i in range(lags,len(data)):
            x_train.append(data[i-lags:i, 0])
            y_train.append(data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        return x_train, y_train

def normalisationPreparationDonnee(data,lags):
    data=data['cpu'].values
    sc = MinMaxScaler(feature_range = (0, 1))
    #sc=StandardScaler()
    data1= data.reshape(-1,1)
    data2 = sc.fit_transform(data1)
    def preparation_data():
        x_train = []
        y_train = []
        for i in range(lags,len(data2)):
            x_train.append(data2[i-lags:i, 0])
            y_train.append(data2[i, 0])
        return np.array(x_train), np.array(y_train)
    x_train, y_train = preparation_data()
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train


 
# IV - Transformation des données pour entrainement dans les differents modèles

lags=5

x_train, y_train = normalisationPreparationDonnee(x_trainning, lags)
x_test, y_test = normalisationPreparationDonnee(x_testing, lags)


# IV.1 - : Entrainement du modele RNN-LSTM

# IV.1.1 - : Initialising the RNN-LSTM


start=time()
modelLSTM = Sequential()
modelLSTM.add(LSTM(units = 100, return_sequences = True, input_shape = (x_train.shape[1], 1)))
modelLSTM.add(LSTM(units = 128, return_sequences = True))
modelLSTM.add(LSTM(units = 64, return_sequences = True))
modelLSTM.add(LSTM(units = 32, return_sequences = True))
modelLSTM.add(LSTM(units = 64, return_sequences = True))
modelLSTM.add(LSTM(units = 128, return_sequences = True))
modelLSTM.add(LSTM(units = 100))
modelLSTM.add(Dense(units = 1))
modelLSTM.compile(optimizer = 'adam', loss = 'mse',
              metrics=['accuracy'])
print(modelLSTM.summary())

historylstm=modelLSTM.fit(x_train, y_train, epochs = 500,batch_size = 5,
                  validation_split=0.20,verbose=1)

elapsed=time()-start

print('duree totale est de :',elapsed/60)
# 27 minutes

plot_model(modelLSTM,show_shapes=True, to_file='regressor.png')

plot_model(modelLSTM,to_file='regressor.png')

# IV.1.2 - :Plot training & validation accuracy values

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.subplot(1,2,2)
# Plot training & validation loss values
plt.figure(figsize=(18,15))
plt.subplot(1,2,1)
plt.plot(historylstm.history['loss'],'b',label='Entrainement')
plt.plot(historylstm.history['val_loss'],'green',label='Validation')
plt.title('Perte pendant l\ ''entrainement et la validation du modèle RNN-LSTM')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Entrainement', 'Validation'], loc='upper left')


#  IV.1.3 - : Enregistrement et importation  du model

modelLSTM.save('modelRnnLSTMProj.h5') 

#creates a HDF5 file 'my_model.h5' et model = load_model('modelRnnLSTMProjetSecurite.h5')
# pour le detruire del model  # deletes the existing model
modelLSTM = load_model('modelRnnLSTMProj.h5')

#  IV.1.4 - : Calcul des erreurs et Affichage des ajustements du modele 
#  sur les données Entrainement et de Test



def ErreurPrediction(Nommodele,data,prediction ):
    print('La R^2-Squared(r2_score) du modèle {} est : {}'.format(Nommodele,r2_score(data,prediction)))
    print('La MAE modèle {} est de: {} '.format(Nommodele,mean_absolute_error(data,prediction)))
    print('La MSE modèle {} est de: {} '.format(Nommodele,mean_squared_error(data,prediction)))
    print('RMSE est:{}'.format(np.sqrt(mean_squared_error(data,prediction))))
   

def LossAccuracy(nomDonnées,modelLSTM,x_test,y_test):
    score = modelLSTM.evaluate(x_test, y_test,verbose=0)
    erreur=modelLSTM.metrics_names
    print('Les pertes sur les données {} sont: '.format(nomDonnées))
    print("%s: %.2f%%" % (erreur[0], score[0]*100))
    print("%s: %.2f%%" % (erreur[1], score[1]*100))


#  IV.1.5 - : Visualisation des graphiques deux à deux

def Affichage(historylstm,modelLSTM,Nommodele):
    y_pred_train=modelLSTM.predict(x_train)
    y_pred_test=modelLSTM.predict(x_test)
    plt.figure(figsize=(18,7))
    plt.subplot(1,3,1)
    plt.plot(historylstm.history['loss'],'b',label='Entrainement')
    plt.plot(historylstm.history['val_loss'],'green',label='Validation')
    plt.title('Perte Entrainement - Validation {}'.format(Nommodele))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Entrainement', 'Validation'], loc='upper left')
    plt.subplot(1,3,2)
    plt.plot(y_train, color = 'green', label = 'Données réelles en % CPU')
    plt.plot(y_pred_train, color = 'blue', label = 'Prédiction du modèle {}'.format(Nommodele))
    plt.title('Ajustement {} : Données Entrainement'.format(Nommodele))
    plt.xlabel('Temps')
    plt.ylabel('% CPU')
    plt.legend()
    plt.subplot(1,3,3)
    plt.plot(y_test, color = 'green', label = 'Données réelles en % CPU')
    plt.plot(y_pred_test, color = 'blue', label = 'Prédiction du modèle {} '.format(Nommodele))
    plt.title('Ajustement {} :Données test'.format(Nommodele))
    plt.xlabel('Temps')
    plt.ylabel('% CPU')
    plt.legend()
    
def Affichage_Model(modelLSTM,NomDumodele):
    y_pred_train=modelLSTM.predict(x_train)
    y_pred_test=modelLSTM.predict(x_test)
    plt.figure(figsize=(18,7))
    plt.subplot(1,2,1)
    plt.plot(y_train, color = 'green', label = 'Données réelles en % CPU')
    plt.plot(y_pred_train, color = 'blue', label = 'Prédiction du modèle {}'.format(NomDumodele))
    plt.title('Ajustement {} : Données Entrainement'.format(NomDumodele))
    plt.xlabel('Temps')
    plt.ylabel('% CPU')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(y_test, color = 'green', label = 'Données réelles en % CPU')
    plt.plot(y_pred_test, color = 'blue', label = 'Prédiction du modèle {} '.format(NomDumodele))
    plt.title('Ajustement {} :Données test'.format(NomDumodele))
    plt.xlabel('Temps')
    plt.ylabel('% CPU')
    plt.legend()
    
ErreurPrediction("Erreur données Entrainement",y_train,modelLSTM.predict(x_train))
ErreurPrediction("Erreur données test",y_test,modelLSTM.predict(x_test))

LossAccuracy("enrainement",modelLSTM,x_train,y_train)
LossAccuracy("Test", modelLSTM,x_test,y_test)

Affichage(historylstm,modelLSTM,"RNN-LSTM")

Affichage_Model(modelLSTM,"RNN-LSTM")

#  IV.2 - : Entrainement du modèle RNN-GRU 

#  IV.2.1 - : Initialisation du modèle Gru 

start=time()
modelGRU = Sequential()
modelGRU.add(GRU(units = 100, return_sequences = True, input_shape = (x_train.shape[1], 1)))
modelGRU.add(GRU(units = 128, return_sequences = True))
modelGRU.add(GRU(units = 64, return_sequences = True))
modelGRU.add(GRU(units = 32, return_sequences = True))
modelGRU.add(GRU(units = 64, return_sequences = True))
modelGRU.add(GRU(units = 128, return_sequences = True))
modelGRU.add(GRU(units = 100))
modelGRU.add(Dense(units = 1))
modelGRU.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
              metrics=['accuracy'])
print(modelGRU.summary())

historygru=modelGRU.fit(x_train, y_train, epochs = 500, batch_size = 5,
                        validation_split=0.20,verbose=1)

elapsed=time()-start

print('duree totale est de :',elapsed/60)
# 21 minutes

plot_model(modelGRU,show_shapes=True, to_file='regressor.png')


#  IV.2.2 - : Enregistrement et importation  du model

modelGRU.save('modelRnnGRUProj.h5') 

#creates a HDF5 file 'my_model.h5' et model = load_model('modelRnnLSTMProjetSecurite.h5')
# pour le detruire del model  # deletes the existing model
modelGRU = load_model('modelRnnGRUProj.h5')


#  IV.2.3 - : Calcul des erreurs & Plot training & validation accuracy values

ErreurPrediction("Erreur données Entrainement",y_train,modelGRU.predict(x_train))
ErreurPrediction("Erreur données test",y_test,modelGRU.predict(x_test))

LossAccuracy("enrainement",modelGRU,x_train,y_train)
LossAccuracy("Test", modelLSTM,x_test,y_test)
Affichage(historygru,modelGRU,"RNN-GRU")


# IV.3 - : Entrainement du model RNN bidirectionnel

#IV.3.1 - : Initialising AND TRAINING the RNN bidirectionnel


start=time()
modelbidir = Sequential()
modelbidir.add(Bidirectional(LSTM(units = 100, return_sequences = True, input_shape = (x_train.shape[1],1))))
modelbidir.add(Bidirectional(LSTM(units = 128, return_sequences = True)))
modelbidir.add(Bidirectional(LSTM(units = 64, return_sequences = True)))
modelbidir.add(Bidirectional(LSTM(units = 32, return_sequences = True)))
modelbidir.add(Bidirectional(LSTM(units = 64, return_sequences = True)))
modelbidir.add(Bidirectional(LSTM(units = 128, return_sequences = True)))
modelbidir.add(Bidirectional(LSTM(units = 100)))
modelbidir.add(Dense(units = 1))
modelbidir.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
              metrics=['accuracy'])

historybidir=modelbidir.fit(x_train, y_train, epochs = 500, batch_size = 5,
                       validation_split=0.20,verbose=1 )

elapsed=time()-start
print(modelbidir.summary())
print('duree totale est de :',elapsed/60)
# 38 minutes

plot_model(modelbidir,show_shapes=True, to_file='regressor.png')


# IV.3.2 - :  Enregistrement et importation  du model

modelbidir.save('modelRnnBidirectionalProj.h5') 
#creates a HDF5 file 'my_model.h5' et model = load_model('modelRnnLSTMProjetSecurite.h5')
# pour le detruire del model  # deletes the existing model
modelbidir = load_model('modelRnnBidirectionalProj.h5')


# IV.3.3 - : Calcul des erreurs & Plot training & validation loss values

ErreurPrediction("Erreur données Entrainement",y_train,modelbidir.predict(x_train))
ErreurPrediction("Erreur données test",y_test,modelbidir.predict(x_test))

LossAccuracy("enrainement",modelbidir,x_train,y_train)
LossAccuracy("Test", modelbidir,x_test,y_test)
Affichage(historybidir,modelbidir,"RNN-Bidirectionnel")


# IV.4 - : Entrainement du modèle ConvNet1D

#IV.4.1 - : Initialising and training the ConvNet1D # forme (samples,time,features)

start=time()
modelConvNet1D = Sequential()
modelConvNet1D.add(Conv1D(100, kernel_size=1,activation='tanh',input_shape=(x_train.shape[1], 1)))
modelConvNet1D.add(Conv1D(128, kernel_size=1,activation='tanh'))
modelConvNet1D.add(MaxPooling1D(pool_size=2,strides=1,padding='same'))
modelConvNet1D.add(Conv1D(64, kernel_size=1,activation='tanh'))
modelConvNet1D.add(Conv1D(32, kernel_size=1,activation='tanh'))
modelConvNet1D.add(Conv1D(64, kernel_size=1,activation='tanh'))
modelConvNet1D.add(Conv1D(128, kernel_size=1,activation='tanh'))
modelConvNet1D.add(MaxPooling1D(pool_size=2,strides=1,padding='same'))
modelConvNet1D.add(GlobalMaxPooling1D())
#modelConvNet1D.add(Flatten()) # pas besoin si globalMaxPooling1D est deja là
modelConvNet1D.add(Dense(128,activation='tanh'))
modelConvNet1D.add(Dense(100,activation='tanh'))
modelConvNet1D.add(Dense(1))
modelConvNet1D.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
              metrics=['accuracy'])

historyConvNet1D=modelConvNet1D.fit(x_train, y_train, epochs = 10, batch_size = 5,
                           validation_split=0.20,verbose=1)

elapsed=time()-start
print(modelConvNet1D.summary())
print('duree totale est de :',elapsed/60)
# 3 minutes

plot_model(modelConvNet1D,show_shapes=True, to_file='regressor .png')


# IV.4.2 - :  Enregistrement et importation  du model

modelConvNet1D.save('modelConvNet1DProj.h5') 
#creates a HDF5 file 'my_model.h5' et model = load_model('modelRnnLSTMProjetSecurite.h5')
# pour le detruire del model  # deletes the existing model
modelConvNet1D = load_model('modelConvNet1DProj.h5')


# IV.4.3 - : Calcul des erreurs & Plot training & validation loss values

ErreurPrediction("Erreur données Entrainement",y_train,modelConvNet1D.predict(x_train))
ErreurPrediction("Erreur données test",y_test,modelConvNet1D.predict(x_test))

LossAccuracy("enrainement",modelConvNet1D,x_train,y_train)
LossAccuracy("Test", modelConvNet1D,x_test,y_test)
Affichage(historyConvNet1D,modelConvNet1D,"ConvNet1D")


# IV.5 : Modèle ensemble 

# IV.5.1 : Modèle ensemble bidirectionnel et conveNet1D

modelRandomFoP=joblib.load('modelRandomProj.pkl')

def modeleEnsemble(modele1,modele2,data):
    mod=0.5*(modele1.predict(data) + modele2.predict(data)) 
    return mod


mod=0.5*(modelConvNet1D.predict(x_train) + modelbidir.predict(x_train)) 

modtest=0.5*(modelConvNet1D.predict(x_test) + modelbidir.predict(x_test)) 
ErreurPrediction("Erreur données Entrainement",y_train,mod)
ErreurPrediction("Erreur données Entrainement",y_test,modtest)


# Affichage_Model(modelLSTM,NomDumodele)

plt.figure(figsize=(18,7))
plt.subplot(1,2,1)
plt.plot(y_train, color = 'green', label = 'Données réelles en % CPU')
plt.plot(mod, color = 'blue', label = 'Prédiction du modèle Ensemble convNet2D + RNN-Bidiectionnel')
plt.title('Ajustement Données Entrainement')
plt.xlabel('Temps')
plt.ylabel('% CPU')
plt.legend()
plt.subplot(1,2,2)
plt.plot(y_test, color = 'green', label = 'Données réelles en % CPU')
plt.plot(modtest, color = 'blue', label = 'Prédiction du modèle Ensemble convNet2D + RNN-Bidiectionnel')
plt.title('Ajustement :Données test')
plt.xlabel('Temps')
plt.ylabel('% CPU')
plt.legend()

ErreurPrediction("Erreur données test",y_test,mod.predict(x_test))

# IV.5.2 : Modèle ensemble bidirectionnel,  conveNet1D et Random Forest

def TraitementDataModelEnsembleConVBidirRf(data,label,lags):
    df=data
    
    def create_features():# Creates time series features from datetime index
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
        y = df[label]
            
        return X,y
    
    def normalisationPreparationDonnee():
        dat=data[label].values
        X_RF,y=create_features()
        sc = MinMaxScaler(feature_range = (0, 1))
        data1= dat.reshape(-1,1)
        data2 = sc.fit_transform(data1)
        X_RF = sc.fit_transform(X_RF)
        x_train = []
        y_train = []
        x_trainRf=[]
        for i in range(lags,len(data2)):
            x_train.append(data2[i-lags:i, 0])
            y_train.append(data2[i, 0])
            x_trainRf.append(X_RF[i,:])
        return np.array(x_train,dtype=np.float32), np.array(y_train,dtype=np.float32),np.array(x_trainRf,dtype=np.float32)
    
    x_train, y_train,x_trainRf = normalisationPreparationDonnee()
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train,x_trainRf
 
x_train , y_train, x_trainRf= TraitementDataModelEnsembleConVBidirRf(x_trainning,label='cpu',lags=5)
x_test , y_test, x_testRf= TraitementDataModelEnsembleConVBidirRf(x_testing,label='cpu',lags=5)
 
# IV - Importation des differents MODELS

modelRandom=RandomForestRegressor(n_estimators=300,max_depth=10, random_state=0)
modelRandom.fit(x_trainRf, y_train)

modelRandom=joblib.load('modelRandomProj.pkl')
modelConvNet1D = load_model('modelConvNet1DProj.h5')
modelbidir = load_model('modelRnnBidirectionalProj.h5')

y_pred_train_RF=np.array(modelRandom.predict(x_trainRf),dtype=np.float32)
y_pred_train_Conv=modelConvNet1D.predict(x_train )
y_pred_train_Bidir=modelbidir.predict(x_train)
                           
y_pred_test_RF=np.array(modelRandom.predict(x_testRf),dtype=np.float32)
y_pred_test_Conv=modelConvNet1D.predict(x_test)
y_pred_test_Bidir=modelbidir.predict(x_test)

y_pred_train_modelTotal= 0.3*(modelbidir.predict(x_train)+ modelConvNet1D.predict(x_train ) + modelRandom.predict(x_trainRf))
y_pred_train_modelTotal1= 0.5*(modelbidir.predict(x_train)+ modelConvNet1D.predict(x_train ))


ErreurPrediction("Erreur bidir données Entrainement",y_train,y_pred_train_modelTotal)

y_pred_test_modelTotal= 0.3*(modelRandom.predict(x_testRf) + modelConvNet1D.predict(x_test ) + modelRandom.predict(x_test)) 
y_pred_test_modelTotal= 0.3*(y_pred_test_RF + y_pred_test_Conv + y_pred_test_Bidir)
y_pred_test_modelTotal= 0.70*y_pred_test_Conv + 0.30*y_pred_test_Bidir

ErreurPrediction("Erreur bidir données Entrainement",y_train,y_pred_train_Bidir)
ErreurPrediction("Erreur conV données Entrainement",y_train,y_pred_train_Conv)
ErreurPrediction("Erreur RF données Entrainement",y_train,y_pred_train_RF)

ErreurPrediction("Erreur données Entrainement",y_test,modtest)

def flatten(x):
    flattened_x=np.empty((x.shape[0],x.shape[2]))
    for i in range(x.shape[0]):
        flattened_x[i]=x[i,(x.shape[1]-1),:]
    return(flattened_x)

luc=flatten(x_train)

plt.plot(modelRandom.feature_importances_)
plt.show()
modelRandom.score(x_train,y_train)

luc=pd.DataFrame(y_pred_train_RF)[0].values


plt.figure(figsize=(18,7))
plt.subplot(1,2,1)
plt.plot(y_train, color = 'green', label = 'Données réelles en % CPU')
plt.plot(y_pred_train_Bidir, color = 'blue', label = 'Prédiction du modèle Ensemble convNet2D + RNN-Bidiectionnel')
plt.title('Ajustement Données Entrainement')
plt.xlabel('Temps')
plt.ylabel('% CPU')
plt.legend()
plt.subplot(1,2,2)
plt.plot(y_test, color = 'green', label = 'Données réelles en % CPU')
plt.plot(y_pred_test_modelTotal, color = 'blue', label = 'Prédiction du modèle Ensemble convNet2D + RNN-Bidiectionnel')
plt.title('Ajustement :Données test')
plt.xlabel('Temps')
plt.ylabel('% CPU')
plt.legend()
plt.show()

a,b,c=1,1,1

modCov_LSTMBI_Rf= a*modelConvNet1D + b*modelbidir + c*modelRandomFoP


ErreurPrediction("Erreur données Entrainement",y_train,mod)
ErreurPrediction("Erreur données Entrainement",y_test,modtest)



# IV.6 - : Entrainement du modèle hybride ConvNet1D - Bidirectional

#IV.6.1 - : Initialising du modèle hybride ConvNet1D- Bidirectional


# Entrainement du modèle - # forme (samples,time,features)
start=time()
modelConvNet1DBidirect = Sequential()
modelConvNet1DBidirect.add(Conv1D(100, kernel_size=1,activation='tanh',input_shape=(x_train.shape[1], 1)))
modelConvNet1DBidirect.add(Conv1D(128, kernel_size=1,activation='tanh'))
modelConvNet1DBidirect.add(MaxPooling1D(pool_size=2,strides=1,padding='same'))
modelConvNet1DBidirect.add(Conv1D(64, kernel_size=1,activation='tanh'))
modelConvNet1DBidirect.add(Bidirectional(LSTM(units = 128, return_sequences = True)))
modelConvNet1DBidirect.add(Bidirectional(LSTM(units = 64, return_sequences = True)))
modelConvNet1DBidirect.add(Bidirectional(LSTM(units = 32, return_sequences = True)))
modelConvNet1DBidirect.add(Bidirectional(LSTM(units = 64, return_sequences = True)))
modelConvNet1DBidirect.add(Bidirectional(LSTM(units = 128, return_sequences = True)))
modelConvNet1DBidirect.add(Bidirectional(LSTM(units = 100)))
modelConvNet1DBidirect.add(Dense(units = 1))
modelConvNet1DBidirect.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
              metrics=['accuracy'])

historyConvNet1DBidirect=modelConvNet1DBidirect.fit(x_train, y_train, 
                                                    epochs = 500, batch_size = 5,
                                                    validation_split=0.20,verbose=1)

elapsed=time()-start
print(modelConvNet1DBidirect.summary())
print('duree totale est de :',elapsed/60)
# 36 minutes

plot_model(modelConvNet1DBidirect,show_shapes=True, to_file='regressor .png')

# IV.6.2 - :  Enregistrement et importation  du model


modelConvNet1DBidirect.save('modelConvNet1DBidirectProj.h5') 
#creates a HDF5 file 'my_model.h5' et model = load_model('modelRnnLSTMProjetSecurite.h5')
# pour le detruire del model  # deletes the existing model
modelConvNet1DBidirect = load_model('modelConvNet1DBidirectProj.h5')

# IV.6.3 - : Calcul des erreurs & Plot training & validation accuracy values

ErreurPrediction("Erreur données Entrainement",y_train,modelConvNet1DBidirect.predict(x_train))
ErreurPrediction("Erreur données test",y_test,modelConvNet1DBidirect.predict(x_test))

LossAccuracy("enrainement",modelConvNet1DBidirect,x_train,y_train)
LossAccuracy("Test", modelConvNet1DBidirect,x_test,y_test)
Affichage(historyConvNet1DBidirect,modelConvNet1DBidirect,"hybride ConvNet1D-Bidirectionel")

# IV.7 : ENTRAINEMENT MODELE VOTING ET STACKING



# IV.7.1 - : Préparation des données pour le  retard(lags) timesteps 


# IV.7.2 - : Préparation des differents modèles 

def get_modelLSTM():
    modelLSTM = Sequential()
    modelLSTM.add(LSTM(units = 100, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    modelLSTM.add(LSTM(units = 128, return_sequences = True))
    modelLSTM.add(LSTM(units = 64, return_sequences = True))
    modelLSTM.add(LSTM(units = 32, return_sequences = True))
    modelLSTM.add(LSTM(units = 64, return_sequences = True))
    modelLSTM.add(LSTM(units = 128, return_sequences = True))
    modelLSTM.add(LSTM(units = 100))
    modelLSTM.add(Dense(units = 1))
    modelLSTM.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
              metrics=['accuracy'])
    return modelLSTM 

def get_modelGRU():
    modelGRU = Sequential()
    modelGRU.add(GRU(units = 100, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    modelGRU.add(GRU(units = 128, return_sequences = True))
    modelGRU.add(GRU(units = 64, return_sequences = True))
    modelGRU.add(GRU(units = 32, return_sequences = True))
    modelGRU.add(GRU(units = 64, return_sequences = True))
    modelGRU.add(GRU(units = 128, return_sequences = True))
    modelGRU.add(GRU(units = 100))
    modelGRU.add(Dense(units = 1))
    modelGRU.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
              metrics=['accuracy'])
    return modelGRU
    

def get_modelbidir():
    modelbidir = Sequential()
    modelbidir.add(Bidirectional(LSTM(units = 100, return_sequences = True, input_shape = (x_train.shape[1],1))))
    modelbidir.add(Bidirectional(LSTM(units = 128, return_sequences = True)))
    modelbidir.add(Bidirectional(LSTM(units = 64, return_sequences = True)))
    modelbidir.add(Bidirectional(LSTM(units = 32, return_sequences = True)))
    modelbidir.add(Bidirectional(LSTM(units = 64, return_sequences = True)))
    modelbidir.add(Bidirectional(LSTM(units = 128, return_sequences = True)))
    modelbidir.add(Bidirectional(LSTM(units = 100)))
    modelbidir.add(Dense(units = 1))
    modelbidir.compile(optimizer = 'adam', loss = 'mse',
              metrics=['accuracy'])
    return modelbidir

def get_modelConvNet1D():
    modelConvNet1D = Sequential()
    modelConvNet1D.add(Conv1D(100, kernel_size=1,activation='tanh',input_shape=(x_train.shape[1], 1)))
    modelConvNet1D.add(Conv1D(128, kernel_size=1,activation='tanh'))
    modelConvNet1D.add(MaxPooling1D(pool_size=2,strides=1,padding='same'))
    modelConvNet1D.add(Conv1D(64, kernel_size=1,activation='tanh'))
    modelConvNet1D.add(Conv1D(32, kernel_size=1,activation='tanh'))
    modelConvNet1D.add(Conv1D(64, kernel_size=1,activation='tanh'))
    modelConvNet1D.add(Conv1D(128, kernel_size=1,activation='tanh'))
    modelConvNet1D.add(MaxPooling1D(pool_size=2,strides=1,padding='same'))
    modelConvNet1D.add(GlobalMaxPooling1D())#modelConvNet1D.add(Flatten()) # pas besoin si globalMaxPooling1D est deja là
    modelConvNet1D.add(Dense(128,activation='tanh'))
    modelConvNet1D.add(Dense(100,activation='tanh'))
    modelConvNet1D.add(Dense(1))
    modelConvNet1D.compile(optimizer = 'adam', loss = 'mse',
              metrics=['accuracy'])
    return modelConvNet1D


def get_modelann():
    reg = Sequential()
    reg.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
    reg.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu'))
    reg.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu'))
    reg.add(Dense(1))
    reg.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['accuracy'])
    return reg


    
# IV.7.3 - : Application du regressor de keras 


model_lstm=KerasRegressor(build_fn=get_modelLSTM,validation_split=0.20,batch_size = 5,epochs=200,verbose=1)
model_gru=KerasRegressor(build_fn=get_modelGRU,validation_split=0.20,batch_size = 5,epochs=100,verbose=1)
model_bidir=KerasRegressor(build_fn=get_modelbidir,validation_split=0.20,batch_size = 5,epochs=200,verbose=1)
model_conv=KerasRegressor(build_fn=get_modelConvNet1D,validation_split=0.20,batch_size = 5,epochs=200,verbose=1)
model_ann=KerasRegressor(build_fn=get_modelann(),batch_size = 5,epochs=1,verbose=1)

# convertion de KerasRegressor en regressor

model_lstm._estimator_type = "regressor"
model_gru._estimator_type = "regressor"
model_bidir._estimator_type = "regressor"
model_conv._estimator_type = "regressor"
model_ann._estimator_type = "regressor"



"""
estimator=estimator
# IV.7.3.1 - : Application du voting Regressor 
x_train, y_train=normalisationPreparationDonnee(x_train,lags=5)
x_test, y_test=normalisationPreparationDonnee(x_test,lags=5)

start=time()

voting_reg = VotingRegressor( estimators=[('lstm', model_lstm),
                                          ('gru', model_gru),
                                          ('bidir', model_bidir),
                                          ('con',model_conv)]
                            )

voting_reg.fit(x_train, y_train) 
elapsed=time()-start
print('duree totale est de :',elapsed/60)
# duree totale est de : 5.476252822081248
# Enregistrement et importation du modèle en format .pkl pour les modeles sklearn

joblib.dump(voting_reg,'modelvoting_regProj.pkl')

# Importation du modèle

voting_reg=joblib.load('modelvoting_regProj.pkl')

voting_reg.save('modelvoting_regProj.h5') 
#creates a HDF5 file 'my_model.h5' et model = load_model('modelRnnLSTMProjetSecurite.h5')
# pour le detruire del model  # deletes the existing model
voting_reg = load_model('modelvoting_regProj.h5')

# Affichage des erreurs

ErreurPrediction("Erreur données entrainement voting",y_train,voting_reg.predict(x_train))
ErreurPrediction("Erreur données test",y_test,voting_reg.predict(x_test))
Affichage_Model(voting_reg,"modèle voting")

"""
# IV.7.3.2 - : Application du Stacking Regressor 



# IV.7.3.2.1 - : Application du Stacking Regressor  avec Random forest
"""
stacking_reg_RF = StackingRegressor( estimators=[('lstm', model_lstm),
                                          ('gru', model_gru),
                                          ('bidir', model_bidir),
                                          ('con',model_conv)],
                                 final_estimator=RandomForestRegressor(n_estimators=100, random_state=42))
stacking_reg_RF.fit(x_train, y_train)
"""

# A :  STACKING SUR RONDOM FOREST ET REGRESSION LINEAIRE

# A1 : STACKING SUR RONDOM FOREST

start=time()

stacking_reg_RF = StackingRegressor( estimators=[
                                          ('bidir', model_bidir),
                                          ('con',model_conv)],
                                 final_estimator=RandomForestRegressor(n_estimators=300, random_state=42),cv=2)
stacking_reg_RF.fit(x_train, y_train)

elapsedRf=time()-start
print('duree totale est de :',elapsedRf/60)
# duree totale est de : 80.42920960187912
# Enregistrement et importation du modèle en format .pkl pour les modeles sklearn

from keras.models import load_model,save_model

stacking_reg_RF.save('stacking_reg_RF.h5')

save_model(stacking_reg_RF,'stacking_reg_RF.h5')

joblib.dump(stacking_reg_RF,'modelstacking_reg_RFProj.joblib')

dump(stacking_reg_RF, 'modelstacking_reg_RFProj.joblib')

stacking_reg_RF=joblib.load('modelstacking_reg_RFProj.pkl')
stacking_reg_RF = load('modelstacking_reg_RFProj.joblib')


stacking_reg_RF.save('modelstacking_reg_RFProj.h5') 
stacking_reg_RF= load_model('modelstacking_reg_RFProj.h5')


ErreurPrediction("Erreur données entrainement stacking RF",y_train,stacking_reg_RF.predict(x_train))
ErreurPrediction("Erreur données test RF",y_test,stacking_reg_RF.predict(x_test))
Affichage_Model(stacking_reg_RF,"modèle stacking par Forets Aleatoires")

# A2 :  STACKING SUR REGRESSION LINEAIRE

# IV.7.3.2.2 - : Application du Stacking Regressor  avec Regression lineaire

start=time()

stacking_reg_REGL = StackingRegressor( estimators=[('bidir', model_bidir),
                                          ('con',model_conv)],
                                 final_estimator=LinearRegression(),cv=2)

stacking_reg_REGL.fit(x_train, y_train)

elapsedRGL=time()-start

print('duree totale est de :',elapsedRGL/60)
#duree totale pour 100 epoch est de : 127.59349313179652
# Enregistrement et importation du modèle en format .pkl 

"""
import joblib
from joblib import dump, load

"""
joblib.dump(stacking_reg_REGL,'modelstacking_reg_REGLProj.pkl')
dump(stacking_reg_REGL, 'modelstacking_reg_REGLProj.joblib')

stacking_reg_REGL=joblib.load('modelstacking_reg_REGLProj.pkl')
stacking_reg_REGL = load('modelstacking_reg_RFProj.joblib')

"""
ErreurPrediction("Erreur données entrainement stacking REGL",y_train,stacking_reg_REGL.predict(x_train))
ErreurPrediction("Erreur données test REGL",y_test,stacking_reg_REGL.predict(x_test))
Affichage_Model(stacking_reg_REGL,"modèle stacking par Régression Lineaire")

# A3 :  COMPARAISON DE STACKING SUR RONDOM FOREST ET REGRESSION LINEAIRE
"""
dump([stacking_reg_RF, stacking_reg_REGL], 'modelstacking_reg_RF_REGLProj.joblib', compress=1)
stacking_reg_RF, stacking_reg_REGL = load('modelstacking_reg_RF_REGLProj.joblib')

plt.figure(figsize=(18,7))
plt.subplot(1,2,1)
plt.plot(y_train, color = 'green', label = 'Données réelles en % CPU')
plt.plot(stacking_reg_RF.predict(x_train), color = 'blue', label = 'Prédiction modèle stacking par Randon foret')
plt.plot(stacking_reg_REGL.predict(x_train), color = 'black', label = 'Prédiction modèle stacking Regression linéaire')
plt.title('Ajustement Données Entrainement')
plt.xlabel('Temps')
plt.ylabel('% CPU')
plt.legend()
plt.subplot(1,2,2)
plt.plot(y_test, color = 'green', label = 'Données réelles en % CPU')
plt.plot(stacking_reg_RF.predict(x_test), color = 'blue', label = 'Prédiction modèle stacking par Random foret')
plt.plot(stacking_reg_REGL.predict(x_test), color = 'black',marker='*', label = 'Prédiction modèle stacking par Regression linéaire')
plt.title('Ajustement :Données test')
plt.xlabel('Temps')
plt.ylabel('% CPU')
plt.legend()
plt.show()


elapsedFinal=time()-start

print('duree totale est de :',elapsedFinal/60)


joblib.dump(stacking_reg_RF,'modelstacking_reg_RFProj.joblib')


joblib.dump(voting_reg,'modelvoting_regProj.pkl')
import pickle
import json
from keras.models import model_from_json

def save_model(model,nomModel,nomPoids):
    # saving model
    json_model = model.to_json()
    open('nomModel', 'w').write(json_model)
    # saving weights
    model.save_weights('nomPoids.h5', overwrite=True)

def load_model(nomModel,model_weights):
    # loading model
    model = model_from_json(open('nomModel.json').read())
    model.load_weights('model_weights')
    model.compile(loss='mse', optimizer='adam')
    return model

save_model(stacking_reg_RF,'modelstacking_reg_RFProj.json','modelstacking_reg_RFProj_weights.h5')

save_model(stacking_reg_REGL,'modelstacking_reg_REGLProj','modelstacking_reg_REGLProj_weights')


load_model('modelstacking_reg_RFProj','modelstacking_reg_RFProj_weights')
load_model('modelstacking_reg_REGLProj','modelstacking_reg_REGLProj_weights')

joblib.dump(stacking_reg_REGL, 'modelstacking_reg_REGLProj.mod')


dump(stacking_reg_REGL, 'modelstacking_reg_REGLProj.joblib')
dump(stacking_reg_RF, 'modelstacking_reg_RFProj.joblib')

pickle.dump(stacking_reg_RF,'modelstacking_reg_RFProj.pkl','w')
pickle.dump(stacking_reg_REGL,'modelstacking_reg_REGLProj.pkl','w')

dump([stacking_reg_RF, stacking_reg_REGL], 'modelstacking_reg_RF_REGLProj.joblib', compress=1)

stacking_reg_REGL.save('modelstacking_reg_REGLProj.h5') 
stacking_reg_RF.save('modelstacking_reg_RFProj.h5')



save_model(stacking_reg_REGL, 'modelstacking_reg_REGLProj')
save_model(stacking_reg_RF, 'modelstacking_reg_RFProj')



#creates a HDF5 file 'my_model.h5' et model = load_model('modelRnnLSTMProjetSecurite.h5')
# pour le detruire del model  # deletes the existing model
modelConvNet1D = load_model('modelConvNet1DProj.h5')


stacking_reg_REGL=joblib.load('modelstacking_reg_REGLProj.pkl')
stacking_reg_REGL = load('modelstacking_reg_RFProj.joblib')

dump([stacking_reg_RF, stacking_reg_REGL], 'modelstacking_reg_RF_REGLProj.joblib', compress=1)

stacking_reg_RF, stacking_reg_REGL = load('modelstacking_reg_RF_REGLProj.joblib')
# IV.7.3.2.3 - : Application du Stacking Regressor  avec Reseau de neuronne artificiel




start=time()
stacking_reg_xgboost = StackingRegressor( estimators=[                                        
                                          ('bidir', model_bidir),
                                          ('con',model_conv)],
                                 final_estimator=GradientBoostingRegressor(loss='quantile',criterion='mse', alpha=0.9,
                                n_estimators=300, max_depth=20,learning_rate=0.1, min_samples_leaf=3,
                                min_samples_split=20,
                                max_features='log2'), cv=2)
stacking_reg_xgboost.fit(x_train, y_train)
elapsed=time()-start

print('duree totale est de :',elapsed/60)
#duree totale est de : 108.96804370482762
# Enregistrement et importation du modèle

joblib.dump(stacking_reg_xgboost,'modelstacking_reg_xgboostProj.pkl')
stacking_reg_xgboost=joblib.load('modelstacking_reg_xgboostProj.pkl')

ErreurPrediction("Erreur données entrainement stacking XGBoost",y_train,stacking_reg_xgboost.predict(x_train))
ErreurPrediction("Erreur données test",y_test,stacking_reg_xgboost.predict(x_test))
Affichage_Model(stacking_reg_xgboost,"modèle stacking par XGBoost")


   
# IV.8  Approche par les méthodes hybrides : Conv1Net – Auto Encoder - RNN-Bidirectionnel



start=time()

data_input=Input(shape=(x_train.shape[1], 1))

#ConvNet1D=layers.Reshape(( None,x_train.shape[1],1))(data_input)
ConvNet1D=layers.Conv1D(100, kernel_size=1,activation='tanh')(data_input)
ConvNet1D=layers.Conv1D(128, kernel_size=1,activation='tanh')(ConvNet1D)
ConvNet1D=layers.MaxPooling1D(pool_size=2,strides=1,padding='same')(ConvNet1D)
ConvNet1D=layers.Conv1D(64, kernel_size=1,activation='tanh')(ConvNet1D)
ConvNet1D=layers.Conv1D(32, kernel_size=1,activation='tanh')(ConvNet1D)
ConvNet1D=layers.Conv1D(64, kernel_size=1,activation='tanh')(ConvNet1D)
ConvNet1D=layers.Conv1D(128, kernel_size=1,activation='tanh')(ConvNet1D)
ConvNet1D=layers.MaxPooling1D(pool_size=2,strides=1,padding='same')(ConvNet1D)
ConvNet1D=layers.GlobalMaxPooling1D()(ConvNet1D)
#modelConvNet1D.add(Flatten()) # pas besoin si globalMaxPooling1D est deja là
ConvNet1D1=layers.Dense(128,activation='tanh')(ConvNet1D)

print(ConvNet1D1.summary())

ConvNet1D1=layers.Reshape(( -1,5, 128))

ConvNet1D1.target_shape

autoencod=layers.LSTM(100,return_sequences=True)(data_input)
autoencod=layers.LSTM(128, return_sequences=True)(autoencod)
autoencod=layers.LSTM(64,return_sequences=True)(autoencod)
autoencod=layers.LSTM(64,return_sequences=True)(autoencod)
autoencod=layers.LSTM(32,return_sequences=False)(autoencod)
autoencod=layers.RepeatVector(x_train.shape[1])(autoencod)
autoencod=layers.LSTM(128,return_sequences=True)(autoencod)

print(autoencod.summary())

modGRU=layers.GRU(units = 100, return_sequences = True)(data_input)
modGRU=layers.GRU(units = 128, return_sequences = True)(modGRU)
modGRU=layers.GRU(units = 64, return_sequences = True)(modGRU)
modGRU=layers.GRU(units = 32, return_sequences = True)(modGRU)
modGRU=layers.GRU(units = 64, return_sequences = True)(modGRU)
modGRU=layers.GRU(units = 128, return_sequences = True)(modGRU)


modelconcat=layers.concatenate([ConvNet1D1,autoencod,modGRU],axis=1)


modelconcat=layers.concatenate((ConvNet1D,autoencod,modGRU))



modelconcat=layers.concatenate([ConvNet1D1,autoencod,modGRU],axis=-1)

modelconcat=layers.concatenate([autoencod,modGRU],axis=-1)
modelensemble=layers.Bidirectional(LSTM(units = 128, return_sequences = True))(modelconcat)
modelensemble=layers.Bidirectional(LSTM(units = 64, return_sequences = True))(modelensemble)
modelensemble=layers.Bidirectional(LSTM(units = 32, return_sequences = True))(modelensemble)
modelensemble=layers.Bidirectional(LSTM(units = 64, return_sequences = True))(modelensemble)
modelensemble=layers.Bidirectional(LSTM(units = 128, return_sequences = True))(modelensemble)
modelensemble=layers.Bidirectional(LSTM(units = 100))(modelensemble)
modelensembleF=layers.Dense(units = 1)(modelensemble)

model=Model(data_input,modelensembleF)

model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
              metrics=['accuracy'])

historymodel=model.fit(x_train, y_train, epochs = 500, batch_size = 5,validation_split=0.20,verbose=1)


elapsed=time()-start
print('duree totale est de(en seconde) :',elapsed)
print('duree totale est de(en minute):',elapsed/60)
#duree totale est de(en seconde) : 7600.530438899994
#duree totale est de(en minute): 126.6755073149999
# Enregistrement et importation du modèle
model.save('modelhybrideProj.h5') 
model= load_model('modelhybrideProj.h5')
                                                 
ErreurPrediction("Erreur données entrainement hybride autoencoder-GRU-Bidir",y_train,model.predict(x_train))
ErreurPrediction("Erreur données test",y_test,model.predict(x_test))
Affichage(historymodel,model,"hybride GRU-AutoEncoder-Bidirectionel")
Affichage_Model(model,"hybride GRU-AutoEncoder-Bidirectionel")
LossAccuracy("entrainement",model,x_train,y_train)
LossAccuracy("Test", model,x_test,y_test)


"""
reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
reg3 = LinearRegression()

reg1.fit(X, y)
reg2.fit(X, y)
reg3.fit(X, y)

ereg = VotingRegressor([('gb', reg1), ('rf', reg2), ('lr', reg3)])
ereg.fit(X, y)
"""
# Affichage des R2

# V : Test de securité des differentes approches


modelLSTM = load_model('modelRnnLSTMProj.h5')
modelGRU = load_model('modelRnnGRUProj.h5')
modelbidir = load_model('modelRnnBidirectionalProj.h5')
modelConvNet1D = load_model('modelConvNet1DProj.h5')


modelConvNet1DBidirect = load_model('modelConvNet1DBidirectProj.h5')
model= load_model('modelhybrideProj.h5')

stacking_reg_RF=joblib.load('modelstacking_reg_RFProj.pkl')


stacking_reg_REGL=joblib.load('modelstacking_reg_REGLProj.pkl')

stacking_reg_xgboost=joblib.load('modelstacking_reg_xgboostProj.pkl')

modelRandomF=joblib.load('modelRandomProj.pkl')

def modeleEnsembleConvBidir(modelConvNet1D,modelbidir,x_train,a,b):
    y_pred_con=modelConvNet1D.predict(x_train)
    y_pred_bidir=modelbidir.predict(x_train)
    mod=a*y_pred_con + b*y_pred_bidir 
    return mod


def modeleEnsembleConvBidirRF(modelConvNet1D,modelbidir,modelRandom,x_train,a,b,c):
    y_pred_con=modelConvNet1D.predict(x_train)
    y_pred_bidir=modelbidir.predict(x_train)
    y_pred_train_RF=np.array(modelRandom.predict(x_trainRf),dtype=np.float32).reshape(-1,1)
    y_pred=a*y_pred_con + b*y_pred_bidir + c*y_pred_train_RF
    return  y_pred

x_trainning.mean() # 1.26
x_trainning.std() #♠0.49
x_trainning.min() # 0.57
x_trainning.max() # 2.55
len(x_trainning)#561
len(x_testing)#98

def normalisationPreparationDonnee(data,lags):
    #data=data['cpu'].values
    sc = MinMaxScaler(feature_range = (0, 1))
    data1= data.reshape(-1,1)
    data2 = sc.fit_transform(data1)
    def preparation_data():
        x_train = []
        y_train = []
        for i in range(lags,len(data2)):
            x_train.append(data2[i-lags:i, 0])
            y_train.append(data2[i, 0])
        return np.array(x_train), np.array(y_train)
    x_train, y_train = preparation_data()
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train

# Nous allons generer 100 observations d' une distribution gaussienne de moyenne 1.26 et d'ecart type =0.49 
# une distribution uniform entre le minimum :0.57 et le le maximum:2.55 

x_trainNormale=np.random.normal(1.26, 0.49, 100) # : une array de 7 valeurs issues d'une loi normale de moyenne 5 et écart-type 2.
x_trainUniform=np.random.uniform(0.57, 2.55, 100)# : une array de 7 valeurs issues d'une loi uniforme entre 0 et 2.

"""
x_t=np.random.randn(10)#  : array 1d de 10 nombres d'une distribution gaussienne standard (moyenne 0, écart-type 1).

x_tr=np.random.randint(1, 5, 10) #  : une array 1d de 10 nombres entiers entre 1 et 5, 5 exclus.
x_train1=np.random.standard_normal(7)# : une array de 7 valeurs issues d'une loi normale standard (moyenne 0, écart-type 1).

x_train2=np.random.normal(5, 2, 7) # : une array de 7 valeurs issues d'une loi normale de moyenne 5 et écart-type 2.
x_trainn=np.random.uniform(0, 2, 7)# : une array de 7 valeurs issues d'une loi uniforme entre 0 et 2.

"""

x_train , y_train= normalisationPreparationDonnee(x_trainNormale,lags=5)

y_pred_EnsConBidir=modeleEnsembleConvBidir(modelConvNet1D,modelbidir,x_train,0.45,0.55)
#y_pred_EnsConBidirRF= modeleEnsembleConvBidirRF(modelConvNet1D,modelbidir,modelRandom,x_train,0.43,0.55,0.01)
y_pred_hybrid_convBidir=modelConvNet1DBidirect.predict(x_train)
y_pred_hybrid_GruAutoBidir=model.predict(x_train)

"""
y_pred_stacking_reg_RF=stacking_reg_RF.predict(x_train)
y_pred_stacking_reg_REGL=stacking_reg_REGL.predict(x_train)
y_pred_stacking_reg_xgboost=stacking_reg_xgboost.predict(x_train)
"""

I=[y_pred_EnsConBidir,#y_pred_EnsConBidirRF
   y_pred_hybrid_convBidir,
          y_pred_hybrid_GruAutoBidir,#y_pred_stacking_reg_RF,y_pred_stacking_reg_REGL,
          #y_pred_stacking_reg_xgboost
          ]

for i in I:
    x=[]
    x1=[]
    x.append(mean_squared_error(y_train, i))
    x1.append(r2_score(y_train, i))
    print("mse: {}".format( mean_squared_error(y_train, i,squared=False)))
    print('R2 score: {:.2f}'.format(r2_score(y_train, i)))
 


plt.figure(figsize=(18,15))
plt.subplot(2,2,1)
plt.plot(y_train, color = 'green', label = 'Données réelles en % CPU')
plt.plot(y_pred_EnsConBidir, color = 'blue', label = 'Prédiction du modèle Ensemble convNet2D + RNN-Bidiectionnel')
plt.legend()
plt.subplot(2,2,2)
plt.plot(y_pred_hybrid_convBidir, color = 'blue', label = 'Prédiction du modèle hybride convNet2D + RNN-Bidiectionnel')
plt.plot(y_train, color = 'green', label = 'Données réelles en % CPU')
plt.legend()
plt.subplot(2,2,3)
plt.plot(y_train, color = 'green', label = 'Données réelles en % CPU')
plt.plot(y_pred_hybrid_GruAutoBidir, color = 'red', label = 'Prédiction du modèle hybride GRU + Autoencoder + RNN-Bidiectionnel')
plt.title('Ajustement Données normale')
plt.xlabel('Temps')
plt.ylabel('% CPU')
plt.legend()

def testDeSecuriteModele(data,libele):
    x_train , y_train= normalisationPreparationDonnee(data,lags=5)
    y_pred_EnsConBidir=modeleEnsembleConvBidir(modelConvNet1D,modelbidir,x_train,0.45,0.55)
    #y_pred_EnsConBidirRF= modeleEnsembleConvBidirRF(modelConvNet1D,modelbidir,modelRandom,x_train,0.43,0.55,0.01)
    y_pred_hybrid_convBidir=modelConvNet1DBidirect.predict(x_train)
    y_pred_hybrid_GruAutoBidir=model.predict(x_train)
    
    """
    y_pred_stacking_reg_RF=stacking_reg_RF.predict(x_train)
    y_pred_stacking_reg_REGL=stacking_reg_REGL.predict(x_train)
    y_pred_stacking_reg_xgboost=stacking_reg_xgboost.predict(x_train)
    """
    plt.figure(figsize=(21,11))
    plt.subplot(1,3,1)
    plt.plot(y_train, color = 'green', label = 'Données {}'.format(libele))
    plt.plot(y_pred_EnsConBidir, color = 'blue', marker='v',label = 'Modèle Ensemble(convNet2D + RNN-Bidiectionnel)')
    plt.title('Modèle Ensemble(convN+RNN-Bi. sur Données {}'.format(libele))
    plt.text(5,0.92,'MSE= %2.3f\\ R^2 = %2.3f '%(mean_squared_error(y_train,y_pred_EnsConBidir),
                                                      r2_score(y_train, y_pred_EnsConBidir)),fontsize=12)
    plt.legend()
    ErreurPrediction('Modèle Ensemble(convN+RNN-Bi.)',y_train,y_pred_EnsConBidir )
    plt.subplot(1,3,2)
    plt.plot(y_pred_hybrid_convBidir, color = 'blue',marker='v', label = 'Modèle hybride(convNet2D + RNN-Bidir.)')
    plt.plot(y_train, color = 'green', label = 'Données {}'.format(libele))
    plt.title('Modèle hybride(conv+RNN-Bidir. sur Données {}'.format(libele))
    plt.text(5,0.92,'MSE= %2.3f\\ R^2 = %2.3f '%(mean_squared_error(y_train,y_pred_hybrid_convBidir),r2_score(y_train, y_pred_hybrid_convBidir)))
    plt.legend()
    ErreurPrediction('Modèle hybride(convNet2D + RNN-Bidir.)',y_train,y_pred_hybrid_convBidir )
    plt.subplot(1,3,3)
    plt.plot(y_train, color = 'green', label = 'Données {}'.format(libele))
    plt.plot(y_pred_hybrid_GruAutoBidir, color = 'red',marker='v', label = 'Modèle hybride(GRU + Autoencoder + RNN-Bidir.)')
    plt.title('Modèle hybride(GRU+Autoencoder+RNN-Bi.) sur Données {}'.format(libele))
    plt.text(5,0.92,'MSE= %2.3f\\ R^2 = %2.3f '%(mean_squared_error(y_train,y_pred_hybrid_GruAutoBidir),r2_score(y_train, y_pred_hybrid_GruAutoBidir)))
    plt.xlabel('Temps')
    plt.ylabel('% CPU')
    plt.legend()
    plt.show()
    ErreurPrediction('Modèle hybride(GRU+Autoencoder+RNN-Bi.)',y_train,y_pred_hybrid_GruAutoBidir )


# ErreurPrediction(Nommodele,data,prediction )

x_trainning.mean() # 1.26
x_trainning.std() #0.49
x_trainning.min() # 0.57
x_trainning.max() # 2.55 


x_trainNormale=np.random.normal(1.26, 0.49, 90) # : une array de 7 valeurs issues d'une loi normale de moyenne 5 et écart-type 2.
x_trainUniform=np.random.uniform(0.57, 2.55, 90)#     

x_train_binomiale=np.random.binomial(10, 0.5, 90)# 10 tirage prob=0.5, n=100
x_train_normale_standart=np.random.standard_normal(90)#loi normale standard m=0 , N=1

testDeSecuriteModele(x_trainNormale,'normales')
 
testDeSecuriteModele(x_trainUniform,'Uniform')

testDeSecuriteModele(x_train_binomiale,'binomiale')

testDeSecuriteModele(x_train_normale_standart,'Norm standart')


# Test de securité modele de stacking

def testDeSecuriteModeleSTAC(data,libele):
    x_train , y_train= normalisationPreparationDonnee(data,lags=5)
    
    y_pred_stacking_reg_RF=stacking_reg_RF.predict(x_train)
    y_pred_stacking_reg_REGL=stacking_reg_REGL.predict(x_train)
        
    plt.figure(figsize=(18,7))
    plt.subplot(1,2,1)
    plt.plot(y_train, color = 'green', label = 'Données {}'.format(libele))
    plt.plot(y_pred_stacking_reg_RF, color = 'blue', marker='v',label = 'Modèle stacking RF')
    plt.title('Modèle stacking RFsur Données {}'.format(libele))
    plt.text(20,0.1,'MSE= %2.3f\\ R^2 = %2.3f '%(mean_squared_error(y_train,y_pred_stacking_reg_RF),
                                                 r2_score(y_train, y_pred_stacking_reg_RF)),fontsize=12)
    plt.legend()
    ErreurPrediction('Modèle stacking RF',y_train,y_pred_stacking_reg_RF )
    plt.subplot(1,2,2)
    plt.plot(y_pred_stacking_reg_REGL, color = 'blue',marker='v', label = 'Modèle stacking REGL')
    plt.plot(y_train, color = 'green', label = 'Données {}'.format(libele))
    plt.title('Modèle stacking REGL sur Données {}'.format(libele))
    plt.text(20,0.1,'MSE= %2.3f\\ R^2 = %2.3f '%(mean_squared_error(y_train,y_pred_stacking_reg_REGL),r2_score(y_train, y_pred_stacking_reg_REGL)))
    plt.legend()
    ErreurPrediction('Modèle stacking REGL',y_train,y_pred_stacking_reg_REGL )

    

testDeSecuriteModeleSTAC(x_trainNormale,'normales')
 
testDeSecuriteModeleSTAC(x_trainUniform,'Uniform')

testDeSecuriteModeleSTAC(x_train_binomiale,'binomiale')

testDeSecuriteModeleSTAC(x_train_normale_standart,'Norm standard')


# VI - : Performance du modèle et surveillance du modèle

from datetime import datetime
from tensorflow.keras.models import Model, load_model,Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D,MaxPooling1D,LSTM, GRU,Bidirectional,GlobalMaxPooling1D
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.python.keras.optimizers import TFOptimizer
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV,cross_val_score
from tensorflow.keras.utils import plot_model,model_to_dot
# Entrainement du modèle - # forme (samples,time,features)

joblib.dump(stacking_reg_RF,'modelstacking_reg_RFProj.joblib')

start=time()

# celui-là est tres bon

modelConvNet1DBidirectOP = Sequential()
modelConvNet1DBidirectOP.add(Conv1D(128, kernel_size=1,activation='tanh',input_shape=(x_train.shape[1], 1)))
modelConvNet1DBidirectOP.add(Conv1D(100, kernel_size=1,activation='tanh'))
modelConvNet1DBidirectOP.add(MaxPooling1D(pool_size=3,strides=1,padding='same'))
modelConvNet1DBidirectOP.add(Conv1D(100, kernel_size=1,activation='tanh'))
modelConvNet1DBidirectOP.add(Conv1D(128, kernel_size=1,activation='tanh'))
modelConvNet1DBidirectOP.add(MaxPooling1D(pool_size=3,strides=1,padding='same'))
#modelConvNet1DBidirectOP.add(GlobalMaxPooling1D())
modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 128, return_sequences = True)))
modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 100,return_sequences = True)))
modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 64,return_sequences = True)))
modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 32,return_sequences = True)))
modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 64,return_sequences = True)))
modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 100,dropout=0.1,recurrent_dropout=0.1,return_sequences = True)))
modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 128)))
modelConvNet1DBidirectOP.add(Dense(units = 1)) # 18 minutes 0.92 et 0.83
modelConvNet1DBidirectOP.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
                  metrics=['acc'])   

callbacks_list = [    
    # interrompre quand les ameliorations s'arretent
   
    ModelCheckpoint(
            filepath='modelConvBidirOpProj.h5',
            monitor='val_loss',
            save_best_only=True),# nous allons garder le meilleur modèle)
    TensorBoard(
            
            log_dir='luclog'
            ,
            histogram_freq=1,
            embeddings_freq=1
            ),
        
        ]
#tensorboard --logdir=c:/Users/luc/luclog --reload_multifile=true
#TensorBoard 2.3.0 at http://localhost:6006/
#tensorboard dev upload --logdir \'c:/Users/luc/luclog'
#start=time()

historymodelConvNet1DBidirectOP=modelConvNet1DBidirectOP.fit(x_train, y_train, 
                                                    epochs = 150, batch_size = 5,
                                                    callbacks=callbacks_list,
                                                    validation_split=0.2,verbose=1)

elapsed=time()-start
print(modelConvNet1DBidirectOP.summary())
print('duree totale est de :',elapsed/60)
import graphviz
import pydot_ng
plot_model(modelConvNet1DBidirectOP,to_file='modelConvNet1DBidirectOP1.png')

plot_model(modelConvNet1DBidirectOP)
plot_model(modelConvNet1DBidirectOP,show_shapes=True, to_file='modelConvNet1DBidirectOP.png')
model_to_dot(modelConvNet1DBidirectOP)
# IV.6.2 - :  Enregistrement et importation  du model


modelConvNet1DBidirectOP.save('modelConvNet1DBidirOpProj.h5') 
#creates a HDF5 file 'my_model.h5' et model = load_model('modelRnnLSTMProjetSecurite.h5')
# pour le detruire del model  # deletes the existing model
modelConvNet1DBidirectOP = load_model('modelConvNet1DBidirOpProj.h5')

# IV.6.3 - : Calcul des erreurs & Plot training & validation accuracy values

Affichage(historymodelConvNet1DBidirectOP,modelConvNet1DBidirectOP,'Modèle Hybride(Conv-B)')
Affichage_Model(modelConvNet1DBidirectOP,'Modèle Hybride Conv-Bid')
ErreurPrediction("Erreur données Entrainement",y_train,modelConvNet1DBidirectOP.predict(x_train))
ErreurPrediction("Erreur données test",y_test,modelConvNet1DBidirectOP.predict(x_test))
LossAccuracy("enrainement",modelConvNet1DBidirectOP,x_train,y_train)
LossAccuracy("Test", modelConvNet1DBidirectOP,x_test,y_test)

# V : Recherche des meilleurs paramettres du modèle


#  : Recherche du meilleurs du  modèle (optimizer, batch_size)

start=time()


def build_regressor(optimizer):
    modelConvNet1DBidirectOP = Sequential()
    modelConvNet1DBidirectOP.add(Conv1D(400, kernel_size=1,activation='tanh',input_shape=(x_train.shape[1], 1)))
    modelConvNet1DBidirectOP.add(Conv1D(200, kernel_size=1,activation='tanh'))
    modelConvNet1DBidirectOP.add(MaxPooling1D(pool_size=3,strides=1,padding='same'))
    modelConvNet1DBidirectOP.add(Conv1D(100, kernel_size=1,activation='tanh'))
    modelConvNet1DBidirectOP.add(Conv1D(128, kernel_size=1,activation='tanh'))
   
    modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 128, return_sequences = True)))
    modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 100,return_sequences = True)))
    modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 64,return_sequences = True)))
    modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 32,return_sequences = True)))
    modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 100,return_sequences = True)))
    modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 128,return_sequences = True)))
    modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 128)))
    #modelConvNet1DBidirectOP.add(Dense(units = 100,activation='tanh'))
    modelConvNet1DBidirectOP.add(Dense(units = 1)) # 0.94 et 0.81 18 min
    modelConvNet1DBidirectOP.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
                  metrics=['acc'])

    return modelConvNet1DBidirectOP

model=KerasRegressor(build_fn=build_regressor,verbose=1)
parameters={"optimizer":['rmsprop'],
            "epochs":[100,150],
            "batch_size":5
            }

grid_search=GridSearchCV(estimator=model , param_grid=parameters, scoring=make_scorer(mean_squared_error),cv=5)

grid_search=grid_search.fit(x_train,y_train)

best_parameters=grid_search.best_params_
best_score=grid_search.best_score_

elapsed=time()-start

print(" meilleurs score:",best_score)
print(" meilleurs parametres:",best_parameters)
print(" Durée Totale:",elapsed/60)

# VI : Validation du modèle par validation croisée

def build_ConvNet1DBidir():
    modelConvNet1DBidirectOP = Sequential()
    modelConvNet1DBidirectOP.add(Conv1D(128, kernel_size=1,activation='tanh',input_shape=(x_train.shape[1], 1)))
    modelConvNet1DBidirectOP.add(Conv1D(100, kernel_size=1,activation='tanh'))
    modelConvNet1DBidirectOP.add(MaxPooling1D(pool_size=3,strides=1,padding='same'))
    modelConvNet1DBidirectOP.add(Conv1D(100, kernel_size=1,activation='tanh'))
    modelConvNet1DBidirectOP.add(Conv1D(128, kernel_size=1,activation='tanh'))
    modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 128, return_sequences = True)))
    modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 100,return_sequences = True)))
    modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 64,return_sequences = True)))
    modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 32,return_sequences = True)))
    modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 64,return_sequences = True)))
    modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 100,return_sequences = True)))
    modelConvNet1DBidirectOP.add(Bidirectional(LSTM(units = 128)))
    modelConvNet1DBidirectOP.add(Dense(units = 1)) # 18 minutes 0.92 et 0.83
    modelConvNet1DBidirectOP.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
                      metrics=['acc'])     
    return modelConvNet1DBidirectOP

regressoroptimiser=KerasRegressor(build_fn=build_ConvNet1DBidir,batch_size=5,epochs=150)

precision=cross_val_score(estimator=regressoroptimiser,X=x_train,y=y_train,cv=10)
precision.mean()
precision.std()

print('precision:',precision)

"""
ici c'est l'ancien

precision=[-0.00120841, -0.00632367, -0.00748501, -0.001921 ,  -0.00025073, -0.00838645,
 -0.00582629, -0.00436221, -0.00793974, -0.02732443]

"""

precision: [-0.00067296 -0.00982266 -0.00273582 -0.0026288  -0.00940309 -0.00273803
 -0.01292507 -0.01369902 -0.00084973 -0.0086394 ]

# duree totale est de : 579.2400102257728 soit 9 heures

plt.figure(figsize=(18,7))
plt.plot(precision, color = 'blue',#label = 'performance mse du Modèle hybride(convNet1D + RNN-Bidir.)')
         )
plt.text(1.8,-0.008,'$\mu=%.4f,\\sigma=%.4f$'%(precision.mean(),precision.std()))
plt.title('Performance(MSE) Modèle Hybride ConvNet1D + RNN-BIDIRECTIONNELLSTM par validation croisée')
plt.xlabel('10-folds')
plt.ylabel('MSE')
plt.legend()
plt.show()

print(stacking_reg_REGL)

print('duree totale est de :',elapsed/60)

# VII - Detection des incidents d'anomalies 

# Importation des modèles





modelConvNet1DBidirectOP = load_model('modelConvNet1DBidirOpProj.h5')



# 11.2 - Graphique de Distribution des erreurs de prédiction

def affichageDistributionErreurstack(modele,x_train,x_test,y_test,y_train,nomMODEL):
    y_pred_x_train=modele.predict(x_train)
    y_pred_x_test=modele.predict(x_test)
    erreurs1=y_train-y_pred_x_train.reshape(-1,1)
    erreurs2=y_test- y_pred_x_test.reshape(-1,1)
    plt.figure(figsize=(18,7)) 
    plt.subplot(2,2,1)
    sns.histplot(erreurs1,bins=50,kde=True)
    plt.title('{} : Distribution des erreurs - Données Entrainement'.format(nomMODEL))
    plt.subplot(2,2,2)
    sns.histplot(erreurs2,bins=50,kde=True)
    plt.title('{} : Distribution des erreurs -  Données Test'.format(nomMODEL))
    plt.show()
    
  
    
def afficheAnomalie(modele,x_train,y_train,x_test,y_test,THRESHOLD_Inf,THRESHOLD_Sup,nomMODEL):
    
    y_pred_x_train=modele.predict(x_train)
    y_pred_x_test=modele.predict(x_test)
    # Calcul des erreurs
    erreurs1=y_train-y_pred_x_train.reshape(-1,1)
    erreurs2=y_test-y_pred_x_test.reshape(-1,1)
    
    y_pred_x_train=y_pred_x_train.reshape(-1,1)
    y_pred_x_test=y_pred_x_test.reshape(-1,1)
    
    # Traintement données entrainements
    data_score_df_train=np.concatenate((erreurs1,y_train,y_pred_x_train),axis=1)
    data_score_df_train=pd.DataFrame(data_score_df_train,columns=['loss','cpu','cpu_predit'])
    data_score_df_train['THRESHOLD_Inf']=THRESHOLD_Inf
    data_score_df_train['THRESHOLD_Sup']=THRESHOLD_Sup
    data_score_df_train.loc[data_score_df_train['loss']<=THRESHOLD_Inf,'anomalie']=True
    data_score_df_train.loc[data_score_df_train['loss']>=THRESHOLD_Sup,'anomalie']=True
    anomalies_train=data_score_df_train[data_score_df_train.anomalie==True]
    
    # Traintement données Test
    data_score_df_test=np.concatenate((erreurs2,y_test,y_pred_x_test),axis=1)
    data_score_df_test=pd.DataFrame(data_score_df_test,columns=['loss','cpu','cpu_predit'])
    data_score_df_test['THRESHOLD_Inf']=THRESHOLD_Inf
    data_score_df_test['THRESHOLD_Sup']=THRESHOLD_Sup
    data_score_df_test.loc[data_score_df_test['loss']<=THRESHOLD_Inf,'anomalie']=True
    data_score_df_test.loc[data_score_df_test['loss']>=THRESHOLD_Sup,'anomalie']=True
    anomalies_test=data_score_df_test[data_score_df_test.anomalie==True]
    
    plt.figure(figsize=(18,7))
    plt.subplot(1,2,1)
    plt.plot(range(0,len(data_score_df_train.index)),data_score_df_train.cpu,'blue',label='cpu d''origine')
    plt.plot(range(0,len(data_score_df_train.index)),data_score_df_train.cpu_predit,marker='v',label='prediction CPU(%)')
    sns.scatterplot(anomalies_train.index,anomalies_train.cpu,color=sns.color_palette()[3], s=300,label='anomalie')
    plt.xticks(rotation=25)
    plt.xlabel('Minutes,heures, jour')
    plt.ylabel('% cpu')
    plt.title('{} : Données Entrainement: {}  anomalies detectées au seuil α ϵ ] {} ; {} ['.format(nomMODEL,anomalies_train.shape[0], THRESHOLD_Inf,THRESHOLD_Sup))
    plt.legend(loc='upper left')
    
    plt.subplot(1,2,2)
    plt.plot(range(0,len(data_score_df_test.index)),data_score_df_test.cpu,'blue',label='cpu d''origine')
    plt.plot(range(0,len(data_score_df_test.index)),data_score_df_test.cpu_predit,marker='v',label='prediction CPU(%)')
    sns.scatterplot(anomalies_test.index,anomalies_test.cpu,color=sns.color_palette()[3], s=300,label='anomalie')
    plt.xticks(rotation=25)
    plt.xlabel('Minutes,heures, jour')
    plt.ylabel('% cpu')
    plt.title('{} : Données Test: {}  anomalies detectées au seuil α ϵ ] {} ; {} ['.format(nomMODEL,anomalies_test.shape[0],THRESHOLD_Inf,THRESHOLD_Sup))
    plt.legend(loc='upper left')
    plt.show()
    
  
modelConvNet1DBidirectOP = load_model('modelConvNet1DBidirOpProj.h5')
   
modele=modelConvNet1DBidirectOP

"""
THRESHOLD_Inf=-0.2
THRESHOLD_Sup=0.2

THRESHOLD_Inf=-1
THRESHOLD_Sup=0.0029

mse=0.002908
MSE=0.0148550

THRESHOLD_Inf=-0.5
THRESHOLD_Sup=0.5

THRESHOLD_Inf=-0.1
THRESHOLD_Sup=0.2
"""



# test de securité
x_trainNormale=np.random.normal(1.26, 0.49, 90)

x_train_normale_standart=np.random.standard_normal(90)

x_trainNor,y_trainNor=normalisationPreparationDonnee(x_trainNormale,5)
x_train_nor_standart,y_train_nor_standart=normalisationPreparationDonnee(x_trainNormale,5)



# securité 



def afficheAnomalieSecurite(modele,x_train,y_train,x_test,y_test,THRESHOLD_Inf,THRESHOLD_Sup,nomMODEL):
    y_pred_x_train=modele.predict(x_train)
    y_pred_x_test=modele.predict(x_test)
    # Calcul des erreurs
    erreurs1=y_train.reshape(-1,1)-y_pred_x_train
    erreurs2=y_test.reshape(-1,1)-y_pred_x_test
    y_train=y_train.reshape(-1,1)
    y_test=y_test.reshape(-1,1)
    # Traintement données entrainements
    data_score_df_train=np.concatenate((erreurs1,y_train,y_pred_x_train),axis=1)
    data_score_df_train=pd.DataFrame(data_score_df_train,columns=['loss','cpu','cpu_predit'])
    data_score_df_train['THRESHOLD_Inf']=THRESHOLD_Inf
    data_score_df_train['THRESHOLD_Sup']=THRESHOLD_Sup
    data_score_df_train.loc[data_score_df_train['loss']<=THRESHOLD_Inf,'anomalie']=True
    data_score_df_train.loc[data_score_df_train['loss']>=THRESHOLD_Sup,'anomalie']=True
    anomalies_train=data_score_df_train[data_score_df_train.anomalie==True]
    # Traintement données Test
    data_score_df_test=np.concatenate((erreurs2,y_test,y_pred_x_test),axis=1)
    data_score_df_test=pd.DataFrame(data_score_df_test,columns=['loss','cpu','cpu_predit'])
    data_score_df_test['THRESHOLD_Inf']=THRESHOLD_Inf
    data_score_df_test['THRESHOLD_Sup']=THRESHOLD_Sup
    data_score_df_test.loc[data_score_df_test['loss']<=THRESHOLD_Inf,'anomalie']=True
    data_score_df_test.loc[data_score_df_test['loss']>=THRESHOLD_Sup,'anomalie']=True
    anomalies_test=data_score_df_test[data_score_df_test.anomalie==True]
    
    plt.figure(figsize=(18,7))
    plt.subplot(1,2,1)
    plt.plot(range(0,len(data_score_df_train.index)),data_score_df_train.cpu,'blue',label='cpu d''origine')
    plt.plot(range(0,len(data_score_df_train.index)),data_score_df_train.cpu_predit,marker='v',label='prediction CPU(%)')
    sns.scatterplot(anomalies_train.index,anomalies_train.cpu,color=sns.color_palette()[3], s=300,label='anomalie')
    plt.xticks(rotation=25)
    plt.xlabel('Minutes,heures, jour')
    plt.ylabel('% cpu')
    plt.title('{} : Données Normale: {}  anomalies detectées au seuil α ϵ ] {} ; {} ['.format(nomMODEL,anomalies_train.shape[0], THRESHOLD_Inf,THRESHOLD_Sup))
    plt.legend(loc='upper left')
    
    plt.subplot(1,2,2)
    plt.plot(range(0,len(data_score_df_test.index)),data_score_df_test.cpu,'blue',label='cpu d''origine')
    plt.plot(range(0,len(data_score_df_test.index)),data_score_df_test.cpu_predit,marker='v',label='prediction CPU(%)')
    sns.scatterplot(anomalies_test.index,anomalies_test.cpu,color=sns.color_palette()[3], s=300,label='anomalie')
    plt.xticks(rotation=25)
    plt.xlabel('Minutes,heures, jour')
    plt.ylabel('% cpu')
    plt.title('{} : Données Normale Standard(0,1): {}  anomalies detectées au seuil α ϵ ] {} ; {} ['.format(nomMODEL,anomalies_test.shape[0],THRESHOLD_Inf,THRESHOLD_Sup))
    plt.legend(loc='upper left')
    plt.show()
    
    


def afficheAnomalieSecuriteStack(modele,x_trainNor,y_trainNor,x_train_nor_standart,y_train_nor_standart,THRESHOLD_Inf,THRESHOLD_Sup,nomMODEL):
    y_pred_x_trainN=modele.predict(x_trainNor)
    y_pred_x_testNS=modele.predict(x_train_nor_standart)
    # Calcul des erreurs
    erreurs1=y_trainNor.reshape(-1,1)-y_pred_x_trainN.reshape(-1,1)
    erreurs2=y_train_nor_standart.reshape(-1,1)-y_pred_x_testNS.reshape(-1,1)
    data_score_df_train=np.concatenate((erreurs1,y_trainNor.reshape(-1,1),y_pred_x_trainN.reshape(-1,1)),axis=1)

    # Traintement données entrainements
   
    data_score_df_train=pd.DataFrame(data_score_df_train,columns=['loss','cpu','cpu_predit'])
    data_score_df_train['THRESHOLD_Inf']=THRESHOLD_Inf
    data_score_df_train['THRESHOLD_Sup']=THRESHOLD_Sup
    data_score_df_train.loc[data_score_df_train['loss']<=THRESHOLD_Inf,'anomalie']=True
    data_score_df_train.loc[data_score_df_train['loss']>=THRESHOLD_Sup,'anomalie']=True
    anomalies_train=data_score_df_train[data_score_df_train.anomalie==True]
    # Traintement données Test
    data_score_df_test=np.concatenate((erreurs2,y_train_nor_standart.reshape(-1,1),y_pred_x_testNS.reshape(-1,1)),axis=1)
    data_score_df_test=pd.DataFrame(data_score_df_test,columns=['loss','cpu','cpu_predit'])
    data_score_df_test['THRESHOLD_Inf']=THRESHOLD_Inf
    data_score_df_test['THRESHOLD_Sup']=THRESHOLD_Sup
    data_score_df_test.loc[data_score_df_test['loss']<=THRESHOLD_Inf,'anomalie']=True
    data_score_df_test.loc[data_score_df_test['loss']>=THRESHOLD_Sup,'anomalie']=True
    anomalies_test=data_score_df_test[data_score_df_test.anomalie==True]
    
    plt.figure(figsize=(18,7))
    plt.subplot(1,2,1)
    plt.plot(range(0,len(data_score_df_train.index)),data_score_df_train.cpu,'blue',label='cpu d''origine')
    plt.plot(range(0,len(data_score_df_train.index)),data_score_df_train.cpu_predit,marker='v',label='prediction CPU(%)')
    sns.scatterplot(anomalies_train.index,anomalies_train.cpu,color=sns.color_palette()[3], s=300,label='anomalie')
    plt.xticks(rotation=25)
    plt.xlabel('Minutes,heures, jour')
    plt.ylabel('% cpu')
    plt.title('{} : Données Normale: {}  anomalies detectées au seuil α ϵ ] {} ; {} ['.format(nomMODEL,anomalies_train.shape[0], THRESHOLD_Inf,THRESHOLD_Sup))
    plt.legend(loc='upper left')
    
    plt.subplot(1,2,2)
    plt.plot(range(0,len(data_score_df_test.index)),data_score_df_test.cpu,'blue',label='cpu d''origine')
    plt.plot(range(0,len(data_score_df_test.index)),data_score_df_test.cpu_predit,marker='v',label='prediction CPU(%)')
    sns.scatterplot(anomalies_test.index,anomalies_test.cpu,color=sns.color_palette()[3], s=300,label='anomalie')
    plt.xticks(rotation=25)
    plt.xlabel('Minutes,heures, jour')
    plt.ylabel('% cpu')
    plt.title('{} : Données Normale Standard(0,1): {}  anomalies detectées au seuil α ϵ ] {} ; {} ['.format(nomMODEL,anomalies_test.shape[0],THRESHOLD_Inf,THRESHOLD_Sup))
    plt.legend(loc='upper left')
    plt.show()

THRESHOLD_Inf=-0.6
THRESHOLD_Sup=0.6

# Affichage des  histogramme pour les seuils

affichageDistributionErreurstack(modelConvNet1DBidirectOP,x_train,x_test,y_test,y_train,'Modèle Hybride') 

affichageDistributionErreurstack(stacking_reg_RF,x_train,x_test,y_test,y_train,'stacking') 
   
# Affichage des anomalies

afficheAnomalie(modelConvNet1DBidirectOP,x_train,y_train,x_test,y_test,THRESHOLD_Inf,THRESHOLD_Sup,'Modèle Hybride')


afficheAnomalie(stacking_reg_RF,x_train,y_train,x_test,y_test,THRESHOLD_Inf,THRESHOLD_Sup,'Modèle Stacking')

# Affichage pour la securité


x_trainNormale=np.random.normal(1.26, 0.49, 90)

x_train_normale_standart=np.random.standard_normal(90)

x_trainNor,y_trainNor=normalisationPreparationDonnee(x_trainNormale,5)
x_train_nor_standart,y_train_nor_standart=normalisationPreparationDonnee(x_trainNormale,5)

afficheAnomalieSecurite(modelConvNet1DBidirectOP,x_trainNor,y_trainNor,x_train_nor_standart,y_train_nor_standart,THRESHOLD_Inf,THRESHOLD_Sup,'Modèle Hybride')

afficheAnomalieSecuriteStack(stacking_reg_RF,x_trainNor,y_trainNor,x_train_nor_standart,y_train_nor_standart,THRESHOLD_Inf,THRESHOLD_Sup,'Modèle Stacking')


# VIII -: Analyse des résidus de chaque modèle

modelLSTM = load_model('modelRnnLSTMProj.h5')
modelGRU = load_model('modelRnnGRUProj.h5')


modelbidir = load_model('modelRnnBidirectionalProj.h5')
modelConvNet1D = load_model('modelConvNet1DProj.h5')
modelhybride= load_model('modelhybrideProj.h5')
modelConvNet1DBidirectOP = load_model('modelConvNet1DBidirOpProj.h5')

stacking_reg_RF
stacking_reg_REGL

Residusmodelbidir=y_train.reshape(-1,1)-modelbidir.predict(x_train) 

ResidusmodelConvNet1D=y_train.reshape(-1,1)-modelConvNet1D.predict(x_train) 

ResidusmodelEns50=y_train.reshape(-1,1)-0.5*(modelConvNet1D.predict(x_train)+modelbidir.predict(x_train))
ResidusmodelEns45_55=y_train.reshape(-1,1)-(0.45*modelConvNet1D.predict(x_train) + 0.55*modelbidir.predict(x_train))
Residusmodelhybride=y_train.reshape(-1,1)-modelhybride.predict(x_train)
ResidusmodelConvNet1DBidirectOP=y_train.reshape(-1,1)-modelConvNet1DBidirectOP.predict(x_train)
Residusstacking_reg_RF=y_train-stacking_reg_RF.predict(x_train)
Residusstacking_reg_REGL=y_train-stacking_reg_REGL.predict(x_train)

I=['Residusmodelbidir','ResidusmodelConvNet1D','ResidusmodelEns50','ResidusmodelEns45_55',
   'Residusmodelhybride','ResidusmodelConvNet1DBidirectOP','Residusstacking_reg_RF',
   'Residusstacking_reg_REGL']

for i in I:
    plt.figure(figsize=(18,7)) 
    #plt.subplot(2,2,1)
    sns.histplot(i,bins=50,kde=True)
    plt.title('{} : Distribution des erreurs - Données Entrainement'.format(i))
    plt.show()

plt.figure(figsize=(18,7)) 
plt.subplot(2,2,1)
sns.histplot(Residusstacking_reg_RF,bins=50,kde=True)
plt.title('{} : Distribution des erreurs - Données Entrainement'.format(Residusstacking_reg_RF))
plt.subplot(2,2,2)
sns.distplot(Residusstacking_reg_REGL,bins=50,kde=True)
plt.title('{} : Distribution des erreurs -  Données Test'.format(Residusstacking_reg_REGL))
plt.show()


plt.figure(figsize=(18,11)) 
plt.subplot(4,2,1)    
sns.distplot(Residusstacking_reg_RF,bins=50,kde=True)
plt.title('Distribution des erreurs -  Residus stacking_reg_RF')
plt.subplot(4,2,2) 
sns.distplot(Residusstacking_reg_REGL,bins=50,kde=True)
plt.title('Distribution des erreurs -  stacking_reg_REGL')
plt.subplot(4,2,3) 
sns.distplot(Residusmodelbidir,bins=50,kde=True)
plt.title('Distribution des erreurs -  Residus model bidir')
plt.subplot(4,2,4) 
sns.distplot(ResidusmodelConvNet1D,bins=50,kde=True)
plt.title('Distribution des erreurs -  Residus model ConvNet1D')
plt.subplot(4,2,5) 
sns.distplot(ResidusmodelEns50,bins=50,kde=True)
plt.title('Distribution des erreurs -  Residus model Ens50')
plt.subplot(4,2,6) 
sns.distplot(ResidusmodelEns45_55,bins=50,kde=True)
plt.title('Distribution des erreurs -  Residus model Ens45_55')
plt.subplot(4,2,7) 
sns.distplot(Residusmodelhybride,bins=50,kde=True)
plt.title('Distribution des erreurs -  Residus model hybride')
plt.subplot(4,2,8) 
sns.distplot(ResidusmodelConvNet1DBidirectOP,bins=50,kde=True)
plt.title('Distribution des erreurs -  Residusmodel ConvNet1DBidirectOP')
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.figure(figsize=(18,15)) 
plt.subplot(4,2,1)  
plot_acf(Residusstacking_reg_RF,title="Autocorrelation Residus stacking RF") 
plt.subplot(4,2,2)
plot_acf(Residusstacking_reg_REGL,title="Autocorrelation Residus stacking RGL") 
plot_acf(ResidusmodelConvNet1D,title="Autocorrelation Residus ConvNet1D")
plot_acf(ResidusmodelConvNet1DBidirectOP,title="Autocorrelation Residus ConvNet1DBid")
plot_acf(Residusmodelhybride,title="Autocorrelation Residus hybride")
plot_acf(Residusmodelbidir,title="Autocorrelation Residus bidir")
plot_acf(ResidusmodelEns50,title="Autocorrelation Residus Ens50")
plot_acf(ResidusmodelEns45_55,title="Autocorrelation Residus Ens45_55")
plt.show()


ResidusmodelEns50=y_train.reshape(-1,1)-0.5*(modelConvNet1D.predict(x_train)+modelbidir.predict(x_train))
ResidusmodelEns45_55=y_train.reshape(-1,1)-(0.45*modelConvNet1D.predict(x_train) + 0.55*modelbidir.predict(x_train))
Residusmodelhybride=y_train.reshape(-1,1)-modelhybride.predict(x_train)
ResidusmodelConvNet1DBidirectOP=y_train.reshape(-1,1)-modelConvNet1DBidirectOP.predict(x_train)
Residusstacking_reg_RF=y_train-stacking_reg_RF.predict(x_train)
Residusstacking_reg_REGL=y_train-stacking_reg_REGL.predict(x_train)





plt.figure(figsize=(18,15)) 
plt.subplot(4,2,1)  
plt.plot(Residusstacking_reg_RF,color="b", label = 'residus') 
plt.plot(pd.DataFrame(Residusstacking_reg_RF).rolling(window = 12).mean(), color = 'red', label = 'Moyenne mobile')
plt.plot(pd.DataFrame(Residusstacking_reg_RF).rolling(window = 12).std(), color = 'black', label = 'Ecart-type mobile')
plt.legend(loc = 'best')
plt.title('Distribution Residus stacking_reg_RF')
plt.subplot(4,2,2) 
sns.distplot(Residusstacking_reg_RF,bins=50,kde=True)
plt.title('Histogramme Residus stacking_reg_RF')
plt.subplot(4,2,3) 
plt.plot(Residusstacking_reg_REGL,color="b", label = 'residus') 
plt.plot(pd.DataFrame(Residusstacking_reg_REGL).rolling(window = 12).mean(), color = 'red', label = 'Moyenne mobile')
plt.plot(pd.DataFrame(Residusstacking_reg_REGL).rolling(window = 12).std(), color = 'black', label = 'Ecart-type mobile')
plt.legend(loc = 'best')
plt.title('Distribution Residus stacking_reg_REGL')
plt.subplot(4,2,4)  
sns.distplot(Residusstacking_reg_REGL,bins=50,kde=True) 
plt.title('Distribution Residus stacking_reg_RGL')
plt.subplot(4,2,5) 
plt.plot(Residusmodelbidir,color="b", label = 'residus') 
plt.plot(pd.DataFrame(Residusmodelbidir).rolling(window = 12).mean(), color = 'red', label = 'Moyenne mobile')
plt.plot(pd.DataFrame(Residusmodelbidir).rolling(window = 12).std(), color = 'black', label = 'Ecart-type mobile')
plt.legend(loc = 'best')
plt.title('Distribution Residus RNN-LSTM(Bidirect)')
plt.subplot(4,2,6)  
sns.distplot(Residusmodelbidir,bins=50,kde=True) 
plt.title('Histogramme Residus RNN-LSTM(Bidirect)')
plt.subplot(4,2,7) 
plt.plot(ResidusmodelConvNet1D,color="b", label = 'residus') 
plt.plot(pd.DataFrame(ResidusmodelConvNet1D).rolling(window = 12).mean(), color = 'red', label = 'Moyenne mobile')
plt.plot(pd.DataFrame(ResidusmodelConvNet1D).rolling(window = 12).std(), color = 'black', label = 'Ecart-type mobile')
plt.legend(loc = 'best')
plt.title('Distribution Residus ConvNet1D')
plt.subplot(4,2,8)  
sns.distplot(ResidusmodelConvNet1D,bins=50,kde=True) 
plt.title('Histogramme Residus ConvNet1D')
plt.show()


plt.figure(figsize=(18,15))
plt.subplot(4,2,1) 
plt.plot(ResidusmodelEns50,color="b", label = 'residus') 
plt.plot(pd.DataFrame(ResidusmodelEns50).rolling(window = 12).mean(), color = 'red', label = 'Moyenne mobile')
plt.plot(pd.DataFrame(ResidusmodelEns50).rolling(window = 12).std(), color = 'black', label = 'Ecart-type mobile')
plt.legend(loc = 'best')
plt.title('Distribution Residus modèle Ens50')
plt.subplot(4,2,2)  
sns.distplot(ResidusmodelEns50,bins=50,kde=True)
plt.title('Histogramme Residus Ens50')
plt.subplot(4,2,3) 
plt.plot(ResidusmodelEns45_55,color="b", label = 'residus') 
plt.plot(pd.DataFrame(ResidusmodelEns45_55).rolling(window = 12).mean(), color = 'red', label = 'Moyenne mobile')
plt.plot(pd.DataFrame(ResidusmodelEns45_55).rolling(window = 12).std(), color = 'black', label = 'Ecart-type mobile')
plt.legend(loc = 'best')
plt.title('Distribution Residus modèle Ens45_55')
plt.subplot(4,2,4)  
sns.distplot(ResidusmodelEns45_55,bins=50,kde=True)
plt.title('Histogramme Residus Ens45_55')
plt.subplot(4,2,5) 
plt.plot(Residusmodelhybride,color="b", label = 'residus') 
plt.plot(pd.DataFrame(Residusmodelhybride).rolling(window = 12).mean(), color = 'red', label = 'Moyenne mobile')
plt.plot(pd.DataFrame(Residusmodelhybride).rolling(window = 12).std(), color = 'black', label = 'Ecart-type mobile')
plt.legend(loc = 'best')
plt.title('Distribution Residus modèle hybride')
plt.subplot(4,2,6)  
sns.distplot(Residusmodelhybride,bins=50,kde=True)
plt.title('Histogramme Residus modèle hybride')
plt.subplot(4,2,7)  
plt.plot(ResidusmodelConvNet1DBidirectOP,color="b", label = 'residus') 
plt.plot(pd.DataFrame(ResidusmodelConvNet1DBidirectOP).rolling(window = 12).mean(), color = 'red', label = 'Moyenne mobile')
plt.plot(pd.DataFrame(ResidusmodelConvNet1DBidirectOP).rolling(window = 12).std(), color = 'black', label = 'Ecart-type mobile')
plt.legend(loc = 'best')
plt.title('Distribution Residus modèle Conv+Bidir')
plt.subplot(4,2,8) 
sns.distplot(ResidusmodelConvNet1DBidirectOP,bins=50,kde=True)
plt.title('Histogramme Residus Conv+Bidir')
plt.show()






I=['Residusmodelbidir','ResidusmodelConvNet1D','ResidusmodelEns50','ResidusmodelEns45_55',
   'Residusmodelhybride','ResidusmodelConvNet1DBidirectOP','Residusstacking_reg_RF',
   'Residusstacking_reg_REGL']


