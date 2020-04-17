import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.model_selection
numeros = sklearn.datasets.load_digits()
imagenes = numeros['images']  # Hay 1797 digitos representados en imagenes 8x8
n_imagenes = len(imagenes)
X = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
Y = numeros['target']
print(np.shape(X), np.shape(Y))
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.5)
scaler = sklearn.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
F1s_Test=[]
Losses=[]
F1s_Train=[]
for i in range(20):
    j=i+1
    mlp = sklearn.neural_network.MLPClassifier(activation='logistic', hidden_layer_sizes=(j),max_iter=1000)
    mlp.fit(X_train, Y_train)
    Losses.append(mlp.loss_)
    F1s_Train.append(sklearn.metrics.f1_score(Y_train, mlp.predict(X_train), average='macro'))
    F1s_Test.append(sklearn.metrics.f1_score(Y_test, mlp.predict(X_test), average='macro'))
    
neuronas=np.arange(1,21)
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.title("Loss")
plt.plot(neuronas,Losses)

plt.subplot(1,2,2)
plt.title("F1")
plt.plot(neuronas,F1s_Train,label='Train')
plt.plot(neuronas,F1s_Test,label='Test')
plt.legend()
plt.savefig('loss_f1.png')
#Elegimos 6

modelo = sklearn.neural_network.MLPClassifier(activation='logistic', hidden_layer_sizes=(6),max_iter=100)
mlp.fit(X_train, Y_train)

plt.figure(figsize=(15,5))


for i in range(6):
    scale = np.max(mlp.coefs_[0])
    plt.subplot(3,2,i+1)
    plt.imshow(mlp.coefs_[0][:,i].reshape(8,8),cmap=plt.cm.RdBu, 
                   vmin=-scale, vmax=scale)
plt.savefig('neuronas.png')      
    