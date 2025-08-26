import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import numpy as np
import cv2

path = 'cat_dog_2.zip'
zip_object = zipfile.ZipFile(file=path, mode='r')
zip_object.extractall('./')
zip_object.close()

training_path = './cat_dog_2/training_set'
test_path = './cat_dog_2/test_set'

# --- GERAÇÃO DE DATASETS E TREINAMENTO ---
gerador_treinamento = ImageDataGenerator(rescale=1./255,
                                         rotation_range=7,
                                         horizontal_flip=True,
                                         zoom_range=0.2)
dataset_treinamento = gerador_treinamento.flow_from_directory(training_path,
                                                             target_size = (64, 64),
                                                             batch_size = 32,
                                                             class_mode = 'categorical',
                                                             shuffle = True)

gerador_teste = ImageDataGenerator(rescale=1./255)
dataset_teste = gerador_teste.flow_from_directory(test_path,
                                                  target_size = (64, 64),
                                                  batch_size = 1,
                                                  class_mode = 'categorical',
                                                  shuffle = False)

# --- CRIAÇÃO E TREINAMENTO DA REDE NEURAL ---
network = Sequential()
network.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation='relu'))
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(Conv2D(32, (3,3), activation='relu'))
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(Flatten())
network.add(Dense(units = 3137, activation='relu'))
network.add(Dense(units = 3137, activation='relu'))
network.add(Dense(units = 2, activation='softmax'))
network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])
historico = network.fit(dataset_treinamento, epochs=10)

# --- AVALIAÇÃO E SALVAMENTO ---
previsoes = network.predict(dataset_teste)
previsoes = np.argmax(previsoes, axis = 1)

from sklearn.metrics import accuracy_score
print(f"Acurácia: {accuracy_score(dataset_teste.classes, previsoes):.2f}")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(dataset_teste.classes, previsoes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Predições')
plt.ylabel('Valores Reais')
plt.show() # Para exibir o gráfico no ambiente local

network.save('modelo_classificador.h5')

# --- TESTE COM UMA NOVA IMAGEM ---
test_image_path = './cat_dog_2/test_set/cat/cat.3500.jpg'
image = cv2.imread(test_image_path)

cv2.imshow('Imagem Original', cv2.resize(image, (200, 200)))
cv2.waitKey(0)
cv2.destroyAllWindows()

image = cv2.resize(image, (64, 64))
image = image / 255
image = image.reshape(-1, 64, 64, 3)

resultado = network.predict(image)
resultado = np.argmax(resultado)

dataset_teste.class_indices
nomes_classes = ['cat', 'dog']

if resultado == 0:
    print(f'A imagem é um(a) {nomes_classes[0]}')
else:
    print(f'A imagem é um(a) {nomes_classes[1]}')
