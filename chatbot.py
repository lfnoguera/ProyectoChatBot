# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 20:18:33 2022

@author: Elkin Vera
"""

import nltk
from nltk.stem import WordNetLemmatizer 
import json 
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

lemmatizer = WordNetLemmatizer()
randomwords = []  
classes = [] 
documents = [] 
ignore_words = ['?', '!'] 
data_file = open('intents.json').read()  
intents = json.loads(data_file)
words = []


for intent in intents['intents']:
    for pattern in intent['patterns']:

        # tomar cada palabra y tokenizarla
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        
        # añadir documentos
        documents.append((w, intent['tag']))

        # añadiendo clases a nuestra lista de clases
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# inicialización de los datos de entrenamiento
training = []
output_empty = [0] * len(classes)

for doc in documents:
    
    # inicialización de la bolsa de palabras
    bag = []
    
    # lista de palabras tokenizadas para el patrón
    pattern_words = doc[0]
    
    # lematizar cada palabra - crear la palabra base, en un intento de representar palabras relacionadas
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # crear nuestra matriz de bolsa de palabras con 1, si la palabra coincide con el patrón actual
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # la salida es un '0' para cada etiqueta y un '1' para la etiqueta actual (para cada patrón)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
    
# barajar nuestras características y convertirlas en np.array
random.shuffle(training)
training = np.array(training)

# crear listas de entrenamiento y de prueba. X - patrones, Y - intentos
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

# Crear modelo - 3 capas. Primera capa 128 neuronas, segunda capa 64 neuronas y tercera capa de salida contiene un número de neuronas
# igual al número de intenciones para predecir la intención de salida con softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilar el modelo. El descenso de gradiente estocástico con gradiente acelerado de Nesterov da buenos resultados para este modelo
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#ajustar y guardar el modelo
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")