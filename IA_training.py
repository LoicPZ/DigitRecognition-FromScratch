#coding: UTF-8
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import random



#On charge le dataset de sklearn
digits =load_digits()
y = digits.target
#On met tout les chiffres en ligne et on transpose 
x = digits.images.reshape((len(digits.images), -1))
#normalisation des données
x=x/x.max()

#On split les données en Entrainement et Test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#il y a 10 chiffres différents possibles on fait un tableau 
#de 10*nombre d'image et on met un 1 dans la case qui correspond
#au chiffre de l'image
Y_train_=np.zeros((10,y_train.shape[0]))
for i in range(y_train.shape[0]):
    Y_train_[y_train[i],i]=1

Y_test_=np.zeros((10,y_test.shape[0]))  
for i in range(y_test.shape[0]):
    Y_test_[y_test[i],i]=1

#Convertions des données en colonnes pour les calcul matriciels
x_train=x_train.T
x_test=x_test.T

#Ici réseau de 64 données x qui vont dans 60 neurones puis 10 en hidden layer puis 10 en sortie
nombre_couche=[x_train.shape[0],60,10]

def initialisation(nombre_couche):

    parametres={}

    L = len(nombre_couche) 
    for l in range(1, L):
        parametres['W' + str(l)] = np.random.randn(nombre_couche[l],nombre_couche[l-1])
        parametres['b' + str(l)] = np.random.randn(nombre_couche[l],1)

    return parametres

def sigmoid_(Z):
    return 1/(1+np.exp(-Z))

def forward_propagation(X, parametres):

    activations = {}
    cache=[]
    A = X
    #On divise par 2 car 2 entrées (une W et une b) par couches
    L = len(parametres) // 2
    for l in range(1,L+1):
        #Le A précédent devient le X de la couche suivante
        A_prev = A 
        #Calcul du polynome
        Z = parametres['W'+str(l)].dot(A_prev) + parametres['b'+str(l)]
        #Fonction d'activation
        A = sigmoid_(Z)
        #On met dans un cache toutes les valeurs de précédente w et b
        cache.append(((A_prev,parametres['W'+str(l)],parametres['b'+str(l)]),A))
        #On met dans un dictionnaire nos activations
        activations['A'+str(l)]=A
    
    return activations,cache

def log_loss(A, Y):
    #m le nombre de données
    m=Y.shape[1]
    #epsilon pour éviter les log(0) qui provoquent des erreurs
    epsilon=1e-15
    cost = (-1/m)*np.sum((Y*np.log(A['A'+str(len(A))]+epsilon)+(1-Y)*np.log(1-A['A'+str(len(A))]+epsilon)))
    cost=np.squeeze(cost)
    return cost

def linear_backward(dZ, cache):
    Z_prev, W, b = cache
    m = Z_prev.shape[1]
    dW = (1/m)*np.dot(dZ,Z_prev.T)
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
    # dA_prev = np.dot(W.T,dZ)
    dZ_prev=np.dot(dW.T,dZ)*Z_prev*(1-Z_prev)
    return dZ_prev, dW, db

def back_propagation(A, y, cache):
    L=len(cache)
    gradient= {}
    #Dérivé des Z final
    dZ=A['A'+str(len(A))]-y
    #On met dans un dictionnaire
    gradient["dZ" + str(L)]=dZ

    #pour chaque couche de neurone
    for l in reversed(range(L)) :
        #on prend les valeurs de la couche x,w,b de la couche actuel
        current_cache=cache[l]
        dZ_prev_temp, dW_temp, db_temp = linear_backward(gradient["dZ"+str(l+1)],current_cache[0])
        gradient["dW" + str(l + 1)] = dW_temp
        gradient["db" + str(l + 1)] = db_temp
        gradient["dZ" + str(l)] = dZ_prev_temp

    return gradient

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-(learning_rate)*grads["dW"+str(l+1)] 
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-(learning_rate)*grads["db"+str(l+1)]
    return parameters

#Fonction principale
def L_layer_model(learning_rate=0.01,num_iterations=10000):
    costs=[]
    accuracy=[]
    predict_accuracy=[]

    parametres=initialisation(nombre_couche)
    #tqdm
    for i in tqdm(range(0,num_iterations)):
        #Propagation avant
        A,cache=forward_propagation(x_train,parametres)
        #Calcul de la fonction cout
        cost=log_loss(A,Y_train_)
        #Calcul des gradients
        grads=back_propagation(A,Y_train_,cache)
        #Mise à jour des paramètres
        parametres=update_parameters(parametres,grads, learning_rate)
        #Enregistrement des données pour créer des courbes
        if i % 500 == 0:
            costs.append(cost)
            accuracy.append(accuracy_score(np.argmax(A['A'+str(len(A))],axis=0),y_train))
            Ap=predict_L_layer(x_test,parametres)
            predict_accuracy.append(accuracy_score(Ap,y_test))
        
    #Affichage de courbe
    plt.figure(figsize=(12,10))
    plt.subplot(2,2,1)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Log loss")
    plt.subplot(2,2,2)
    plt.plot(np.squeeze(accuracy))
    plt.plot(np.squeeze(predict_accuracy))
    plt.ylabel('accuracy training')
    plt.xlabel('iterations')
    plt.title("Train Accuracy")
    plt.show() 
    return parametres

def predict_L_layer(X,parameters):
    prediction=[]
    A,caches=forward_propagation(X,parameters)
    prediction=np.argmax(A['A'+str(len(A))] ,axis=0)
    return prediction

#Fonction d'entrainement du model

parametre=L_layer_model()

#Enregistrement des paramètres du model entrainé
#np.save('mespara',parametre)


#Présentation des digits
# plt.figure(figsize=(12,10))
# for i in range(1,11):
#     plt.subplot(2,5,i)
#     plt.imshow(digits.images[i],'gray')
# plt.show()
#

# #Test sur un test set avec paramètres
# #Chargement des paramètres:
# parametre=np.load('mespara.npy', allow_pickle=True)
# parametre=parametre[()]
# for j in range(15):
#     # i=random.randint(0,len(digits.images))
#     # img=digits.images[i].reshape((64,1)).T
#     # img=img.T
#     predicted_digit=predict_L_layer(x_test,parametre)
#     print('Predicted digit is : '+str(predicted_digit[j]))
#     print('True digit is: '+ str(y_test[j]))
# nberreur=0
# for j in range(len(y_test)):
#     if predicted_digit[j] != y_test[j] : nberreur=nberreur+1
# print("IL y a "+str(nberreur)+" erreurs sur "+str(len(y_test))+" données.")
