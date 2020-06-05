# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 09:47:22 2020

@author: luc

"""

# Partie: 1 - Importation des librairies


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pmdarima as pm
from statsmodels.tsa.seasonal    import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import ndiffs
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import seaborn as sns
from scipy import stats 
# Partie : 2 - Importation des données

Data_training = pd.read_csv("https://raw.githubusercontent.com/oreilly-mlsec/book-resources/599669c7124dffb65ea7f6e0b7626df32496b1d6/chapter3/datasets/cpu-utilization/cpu-full-a.csv", header=0, parse_dates=[0], index_col=0, squeeze=True)

Data_validation = pd.read_csv("https://raw.githubusercontent.com/oreilly-mlsec/book-resources/599669c7124dffb65ea7f6e0b7626df32496b1d6/chapter3/datasets/cpu-utilization/cpu-train-b.csv", header=0, parse_dates=[0], index_col=0, squeeze=True)


# Partie : 3 - Exploration des données
# 3.1 : Visualisation des données
plt.figure(figsize=(18,15)) 
plt.plot(Data_training,color = 'green')
plt.ylabel('CPU %')
plt.xlabel('Dates et heures')
plt.title('Utilisation de la mémoire volatile (CPU) en %')
plt.show()

# 3.2 : Evolution de la moyennne et de l'ecartype de la series

rolling_mean = Data_training.rolling(window = 12).mean()
rolling_std = Data_training.rolling(window = 12).std()
plt.figure(figsize=(18,15)) 
plt.plot(Data_training, color = 'green', label = 'Origine')
plt.plot(rolling_mean, color = 'blue', label = 'Moyenne mobile')
plt.plot(rolling_std, color = 'red', label = 'Ecart-type mobile')
plt.legend(loc = 'best')
plt.title('Moyenne et Ecart-type mobiles')
plt.show()

# 3.3 : Décomposition de la serie

decomposition=seasonal_decompose(Data_training,model='additive',period=48)

fig=plt.figure()
fig=decomposition.plot()
fig.set_size_inches(15,8)


# 3.4 : Test Dickey–Fuller :

result = adfuller(Data_training)
print('Statistiques ADF : {}'.format(result[0]))
print('p-value : {}'.format(result[1]))
print('Valeurs Critiques :')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
    
# 3.5 - Etude des autoccorrelations

fig=plt.figure(figsize=(18,15)) 
ax1 = fig.add_subplot(212)
fig=plot_acf(Data_training, ax=ax1,title="Autocorrelation sur les données origines") 
ax2 = fig.add_subplot(221)
fig=plot_acf(Data_training.diff().dropna(), ax=ax2,title="Autocorrelation sur le 1er ordre de Differenciation")
ax3 = fig.add_subplot(222)
fig=plot_acf(Data_training.diff().diff().dropna(), ax=ax3,title="Autocorrelation sur le 2nd Ordre de Differenciation")
plt.show()

# 4 : Traitement des valeurs extremes

plt.figure(figsize=(18,15))
plt.subplot(2,2,1)
plt.hist(Data_training.cpu)
plt.title("Histogramme & Quartiles")
plt.ylabel("Frequence")
plt.legend(loc='upper right')
plt.subplot(2,2,2)
plt.boxplot(Data_training.cpu)
plt.title("Boite à moustaches")
plt.subplot(2,1,2)
plt.plot(Data_training,color = 'green', label = 'Evolution du cpu(%)')
plt.ylabel('% CPU')
plt.xlabel('Dates et heures')
plt.title('Evolution du cpu(%) au cours du temps ')
plt.legend(loc = 'best')
plt.show()

# Partie : 4 - Modélisation ARIMA(p,k,d)

# 4.1 : Adf Test

p=ndiffs(Data_training, test='adf',max_d=10)  # 2

# KPSS test
p1=ndiffs(Data_training, test='kpss',max_d=10)  # 0

# PP test:
p2=ndiffs(Data_training, test='pp',max_d=10)  # 2

# Test Dickey–Fuller :
result = adfuller(Data_training.diff().dropna())
print('Statistiques ADF : {}'.format(result[0]))
print('p-value : {}'.format(result[1]))
print('Valeurs Critiques :')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))


# 4.2 : Séparation des données d'apprentissage et de test

training_size=int(len(Data_training)*.80)

test_size=len(Data_training)-training_size

training,test=Data_training.iloc[0:training_size],Data_training.iloc[training_size:len(Data_training)]

# verification et affichage


print(training.shape,test.shape)


# 4.3 - Recherche du modèle

# 4.3.1 : Modèle avec saisonnalité

model1 = pm.auto_arima(training, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=12,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

# SARIMAX(1, 0, 1)x(1, 1, 1, 12) avec un AIC = -876.259

modelsaisonalité = sm.tsa.statespace.SARIMAX(training,order=(1, 0, 1),seasonal_order=(1, 1, 1,12),enforce_stationarity=True,enforce_invertibility=True)
print(modelsaisonalité.summary())
modelsaisonalité.plot_diagnostics(figsize=(7,5))
# pas de prise en compte des deux pics

# 4.3.2 : Modèle sans saisonnalité

model2 = pm.auto_arima(training, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=False,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)
print(model2.summary())

#  SARIMAX(2, 1, 2) avec -559.438
# Graphique du Diagnostic modèle ARIMA(2,1,2) avec AIC:-55.4
model2.plot_diagnostics(figsize=(15,10))
plt.show()
# prise en compte de tous les pics des données

# 4.3.3 : Modèle Retenu: sans saisonnalité(ARIMA(2, 1, 2) avec -559.438)

# 4.3.4 : Ajustement du modèle sur les données d'entrainement

model=sm.tsa.ARMA(training, order=(2,1,2)).fit()

# 4.3.5 : Ajustement du modèle sur les données Test
modeltest=sm.tsa.ARMA(test, order=(2,1,2)).fit()

# 4.3.6 : Ajustement du modèle sur les données de validation

modelval=sm.tsa.ARMA(Data_validation, order=(2,1,2)).fit()
modelfull=sm.tsa.ARMA(Data_training, order=(2,1,2)).fit()
# 4.4 : Calcul des erreurs sur les données Test et Validation

print("MAE données test :", mean_absolute_error(test,modeltest.fittedvalues))
print("MSE données test :", mean_squared_error(test,modeltest.fittedvalues))
print("RMSE données test :", sqrt(mean_squared_error(test,modeltest.fittedvalues)))
print("MAE données validation :", mean_absolute_error(Data_validation,modelval.fittedvalues))
print("MSE données validation :", mean_squared_error(Data_validation,modelval.fittedvalues))
print("RMSE données validation :", sqrt(mean_squared_error(Data_validation,modelval.fittedvalues)))

# 4.5 : Ajustements graphiques

plt.figure(figsize=(18,15)) 
ax1 = plt.subplot(221) 
model.plot_predict(dynamic=False,ax=ax1)
plt.xlabel('heures en jours', fontsize=12) 
plt.ylabel('CPU', fontsize=12)
plt.legend(loc = 'best')
plt.title('ARIMA(2,1,2):Données d''entrenement', fontsize=12, fontweight="bold")
ax2 = plt.subplot(222) 
modeltest.plot_predict(dynamic=False,ax=ax2)
plt.xlabel('heures,minutes et jours', fontsize=12)
plt.ylabel('CPU', fontsize=12)
plt.legend(loc = 'best')
plt.title('ARIMA(2,1,2):Données Test', fontsize=12, fontweight="bold")
ax3 = plt.subplot(212) 
modelval.plot_predict(dynamic=False,ax=ax3)
plt.xlabel('heures,minutes et jours', fontsize=12)
plt.ylabel('CPU', fontsize=12)
plt.legend(loc = 'best')
plt.title('ARIMA(2,1,2):Données Validation', fontsize=12, fontweight="bold")
plt.show()

# 4.6 : Etude des Résidus du modèle ARIMA(2,1,2) &  calcul des anomalies les erreures de prediction

# 6 - calcul des anomalies par les erreures de prediction

plt.figure(figsize=(18,15)) 
plt.subplot(2,2,1) 
sns.distplot(model.resid,bins=50,kde=True)
plt.title('Distribution des résidus du Modele ARIMA(2,1,2) sur les données d"entrainement')
plt.legend()

# On observe deux valeurs inf a -0.2 et sup 0.2

THRESHOLD_Inf=-0.2
THRESHOLD_Sup=0.2


 
# 11.3  : Constitution des anomalies et du niveau de seuil de detection 

data_score_df =pd.DataFrame(modeltest.resid,columns=['loss'])
data_score_df['cpu']= test
data_score_df['cpu_predit']=pd.DataFrame(modeltest.predict(),columns=['cpu_predict'])
data_score_df['THRESHOLD_Inf']=THRESHOLD_Inf
data_score_df['THRESHOLD_Sup']=THRESHOLD_Sup

# 11.3.1 : Visualisation du niveau de la courbe des pertes et du seuil

plt.subplot(2,2,2)
plt.plot(data_score_df.index,data_score_df['loss'],label='Résidus(loss) Tests')
plt.plot(data_score_df.index,data_score_df['THRESHOLD_Sup'],label='seuil inferieur')
plt.plot(data_score_df.index,data_score_df['THRESHOLD_Inf'],label='seuill superieur')
plt.xticks(rotation=25)
plt.title('Distribution des erreurs de prédiction et Niveau du seuil sur les Données Tests')
plt.legend()

# 11.3.2 : Tableau des anomalies

data_score_df.loc[data_score_df['loss']<=THRESHOLD_Inf,'anomalie']=True
data_score_df.loc[data_score_df['loss']>=THRESHOLD_Sup,'anomalie']=True


# 11.3.3 : : Visualisation des anomalies sur le graphique 

anomalies=data_score_df[data_score_df.anomalie==True]


plt.subplot(2,2,3)
plt.plot(data_score_df.index,data_score_df.cpu,'blue',label='cpu d''origine');
plt.plot(data_score_df.index,data_score_df.cpu_predit,marker='v',label='prediction sur les données Tests');
sns.scatterplot(
    anomalies.index,anomalies.cpu,
    color=sns.color_palette()[3],
    s=300,
    label='anomalie')
plt.xticks(rotation=25)
plt.xlabel('Minutes,heures, jour') 
plt.ylabel('% cpu') 
plt.title('Nombre d"anomalies au seuil erreur(- 0. et 0.2) sur les  données Tests: %2.0f'%(anomalies.shape[0]))
plt.legend(loc='upper left');


# modelisation avec data_full

full_score_df =pd.DataFrame(modelfull.resid,columns=['loss'])
full_score_df['cpu']= Data_training
full_score_df['cpu_predit']=pd.DataFrame(modelfull.predict(),columns=['cpu_predict'])
full_score_df['THRESHOLD_Inf']=THRESHOLD_Inf
full_score_df['THRESHOLD_Sup']=THRESHOLD_Sup

full_score_df.loc[full_score_df['loss']<=THRESHOLD_Inf,'anomalie']=True
full_score_df.loc[full_score_df['loss']>=THRESHOLD_Sup,'anomalie']=True


# 11.3.3 : : Visualisation des anomalies sur le graphique 

anomaliesfull=full_score_df[full_score_df.anomalie==True]


plt.subplot(2,2,4)
plt.plot(full_score_df.index,full_score_df.cpu,'blue',label='cpu d''origine');
plt.plot(full_score_df.index,full_score_df.cpu_predit,marker='v',label='prediction cpu');
sns.scatterplot(
    anomaliesfull.index,anomaliesfull.cpu,
    color=sns.color_palette()[3],
    s=300,
    label='anomalie')
plt.xticks(rotation=25)
plt.xlabel('Minutes,heures, jour') 
plt.ylabel('% cpu') 
plt.title('Nombre d''anomalies au seuil erreur(- 0.2 et 0.2) données Totale: %2.0f'%(anomaliesfull.shape[0]))
plt.legend(loc='upper left');
plt.show()

print('nombre d''anomalies: ',anomaliesfull.shape[0])
print('nombre d''anomalies: ',anomalies.shape[0])


