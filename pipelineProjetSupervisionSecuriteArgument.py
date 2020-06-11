# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:20:54 2020

@author: luc
"""
import pandas as pd
from cassandra.cluster import Cluster
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from keras.models import load_model
from time import time,sleep
import argparse
from sklearn.metrics import mean_squared_error

# PARTIE N°2: Definition des fonctions et des modules

""" Ici , il faudrait d'abord installer Docker,puis créerer une base des données Cassandra, ensuite créerer une table ou keyspaceet vous importer les données.
https://docs.docker.com/toolbox/toolbox_install_windows/
http://b3d.bdpedia.fr/cassandra_tp.html

"""
# 2.1 :Fonction de connexion à la base des données CASSANDRA

def cassandra():
    def pandas_factory(colnames, rows):
        return pd.DataFrame(rows, columns=colnames)
    try:
        cluster = Cluster(['192.168.99.102'],port=3000)
    except ValueError:
        print('connexion invalide avec cassandra \
              Veuillez verifier le port et l"adresse IP')
    else:
        print('Connexion reuissie à cassandra')
        
    try:
         session = cluster.connect('securite',wait_for_all_pools=True)
    except ValueError:
        print('Verifier le nom de votre table')
    else:
         print('Vous etes maintenant connecté')
    session.row_factory = pandas_factory
    session.default_fetch_size = 300 #
    query = "SELECT* FROM logscpu1"
    rows = session.execute(query)
    df = rows._current_rows 
    return df.sort_values(by='datetime', ascending=False)
# ici, la fonction importe les 300 dernières enregistrements des donnéeses dans cassandra et les converti en fichier DataFrame

# 2.2 :Fonction de Préparation,Modelisation, Detection et Visualisation d'anomalie en temps réelel

def preparation_reshaping_modelisation_data(data,lags,model,n,THRESHOLD_Inf,THRESHOLD_Sup):
    start=time()
    scaler = MinMaxScaler(feature_range = (0, 1))
    data_set= data['cpu'].values.reshape(-1,1)
    data_set_scaled=scaler.fit_transform(data_set)
    x_train = []
    y_train = []
    for i in range(lags,len(data_set_scaled)):
        x_train.append(data_set_scaled[i-lags:i, 0])
        y_train.append(data_set_scaled[i, 0])
    train,y=np.array(x_train), np.array(y_train)
    train=np.reshape(train, (train.shape[0], train.shape[1], n))
    y_pred=scaler.inverse_transform(model.predict(train))
    y_or=scaler.inverse_transform(y.reshape(-1, 1))
    erreurs=y_or-y_pred
    print("Erreur de prevision :%.2f"%(mean_squared_error(y_or,y_pred)*100)+'%') # affichage des erreurs de prévision
    anomalies=pd.DataFrame(index=data[lags:].datetime)
    anomalies['THRESHOLD_Inf']=THRESHOLD_Inf
    anomalies['THRESHOLD_Sup']=THRESHOLD_Sup
    anomalies['cpu_predit']=y_pred
    anomalies['cpu']= y_or
    anomalies['loss']=erreurs
    anomalies.loc[anomalies['loss']<=THRESHOLD_Inf,'anomalie']=True
    anomalies.loc[anomalies['loss']>=THRESHOLD_Sup,'anomalie']=True
    anomalie=anomalies[anomalies.anomalie==True] # identification des anomalies
    elapsed=time()-start 
    print("Durée d'exécution:%.2f"%(elapsed)) # affichage de la durée d'execution de calcul
    plt.figure(figsize=(20,10))
    plt.plot(anomalies.index,anomalies.cpu,'blue',label='cpu réelle'); # affichage cpu réel
    plt.plot(anomalies.index,anomalies.cpu_predit,marker='v',label='prédiction cpu(%)'); # affichage cpu prédite
    sns.scatterplot(anomalie.index,anomalie.cpu,
                    color=sns.color_palette()[3],
                    s=300, label='anomalie')# affichage des anomalies en couleur rouge avec une taille de 300
    plt.xticks(rotation=25)
    plt.xlabel('Minutes,heures, jour')
    plt.ylabel('% cpu')
    plt.title('Le nombre Anomalies detecté en temps réel: %2.0f'%(len(anomalie)))
    plt.legend(loc='upper left')
    plt.show()
# 2.3 :Fonction de Préparation,Modelisation, Detection et Visualisation d'anomalie en temps decallé pour la confirmation
# cette fonction sert à convertir les données de l'autoencoder de trois dimension en format d'origine

def flatten(x):
    flattened_x=np.empty((x.shape[0],x.shape[2]))
    for i in range(x.shape[0]):
        flattened_x[i]=x[i,(x.shape[1]-1),:]
    return(flattened_x)

# fonction pour l'autoencoder - idem que la precedente.

def preparation_reshaping_modelisation_data_autoencoder(data,lags,model,THRESHOLD):
    start=time()
    scaler = MinMaxScaler(feature_range = (0, 1))
    data_set= data['cpu'].values.reshape(-1,1)
    data_set_scaled=scaler.fit_transform(data_set)
    x_train = []
    y_train = []
    for i in range(lags,len(data_set_scaled)):
        x_train.append(data_set_scaled[i-lags:i, 0])
        y_train.append(data_set_scaled[i, 0])
    train,y=np.array(x_train), np.array(y_train)
    train=np.reshape(train, (train.shape[0], train.shape[1], n))
    y_predauto=scaler.inverse_transform(flatten(model.predict(train)))
    y_auto=scaler.inverse_transform(flatten(train))
    Erreurs= y_auto-y_predauto
    print("Erreur de prevision :%.2f"%(mean_squared_error(y_auto,y_predauto)*100)+'%')
    anomalies=pd.DataFrame(index=data[lags:].datetime)
    anomalies['cpu_predit']=y_predauto
    anomalies['cpu']=y_auto
    anomalies['loss']=Erreurs
    anomalies['THRESHOLD']=THRESHOLD
    anomalies['anomalie']=anomalies.loss>anomalies.THRESHOLD
    anomalie=anomalies[anomalies.anomalie==True]
    elapsed=time()-start
    #print("Durée d'exécution(en seconde):%.2f"%(elapsed))
    plt.figure(figsize=(20,10))
    plt.plot(anomalies.index,anomalies.cpu,'blue',label='cpu réelle');
    plt.plot(anomalies.index,anomalies.cpu_predit,marker='v',label='prédiction cpu(%)');
    sns.scatterplot(anomalie.index,anomalie.cpu,
                    color=sns.color_palette()[3],
                    s=300, label='anomalie')
    plt.xticks(rotation=25)
    plt.xlabel('Minutes,heures, jour')
    plt.ylabel('% cpu')
    plt.title('Le nombre Anomalies detecté par AutoEndoder: %2.0f'%(len(anomalie)))
    plt.legend(loc='upper left')
    plt.show()
    print("Durée d'exécution(en seconde):%.2f"%(elapsed))
   
# PARTIE N°3: Pipeline

lags,n=5,1 # 
THRESHOLD_Inf,THRESHOLD_Sup =-0.2 ,0.2  # seuil pour le modèle temps réel
THRESHOLD=0.1 # seuil pour l'autEncoder
# Definition des arguments: nous avons choisi deux argument dont un par defaut c'est-a-dire modèlele temps reel
# le pipeline telecharge ou met àjours jour le modèle tous les 7 jours
ap=argparse.ArgumentParser()
ap.add_argument("-a","--methode",type=str,default="ModelTempReel",
                choices=["ModelTempReel","autoencoder"],
                help="metre ModelTempReel ou autoencoder")
args=vars(ap.parse_args())

if args["methode"]=="autoencoder":
    while(1):
        modelAuto = load_model('modelAutoEncoderProjet.h5') # import du modèle autoencoder
        while(1):
            df=cassandra() # connexion et importation des données 
            preparation_reshaping_modelisation_data_autoencoder(df,lags,modelauto,THRESHOLD)
            sleep(10)#pour la connexion toutes les 10 secondes
        sleep(604800) #pour la maintenance tous les 7 jours soit 24*60*60*7
else:
     while(1):
        model = load_model('modelRnnLSTMProjetSecurite.h5') # import du modèle temps réel
        while(1):
            df=cassandra() # connexion et importation des données 
            preparation_reshaping_modelisation_data(df,lags,model,n,THRESHOLD_Inf,THRESHOLD_Sup)
            sleep(5) # mise àjours toutes les 5 secondes
        sleep(604800) #pour la maintenance tous les 7 jours soit 24*60*60*7
       