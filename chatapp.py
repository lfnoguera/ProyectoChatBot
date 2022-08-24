# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 08:37:41 2022

@author: Elkin Vera
"""

import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import tkinter
from tkinter import *

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# devuelve la matriz de palabras de la bolsa: 0 o 1 para cada palabra de la bolsa que existe en la frase
def bow(sentence, words, show_details=True):
    # tokenizar el patrón
    sentence_words = clean_up_sentence(sentence)
    
    # bolsa de palabras - matriz de N palabras, matriz de vocabulario
    bag = [0]*len(words)
    
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                
                # asigna 1 si la palabra actual está en la posición del vocabulario
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
                    
    return(np.array(bag))

def predict_class(sentence, model):
    # filtrar las predicciones por debajo de un umbral
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    
    # ordenar por fuerza de la probabilidad
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
        
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

#Creación de GUI con tkinter
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Bienvenido al ChatBot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Crear una ventana de chat
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Unir la barra de desplazamiento a la ventana del chat
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Crear un botón para enviar un mensaje
SendButton = Button(base, font=("verdana",12,'bold'), text="Enviar \n mensaje", width="8", height=5,
                    bd=0, bg="#798e93", activebackground="#b4bfbf",fg='#ffffff',
                    command= send )

#Crear la caja para introducir el mensaje
EntryBox = Text(base, bd=0, bg="#dbd5c5",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Coloca todos los componentes en la pantalla
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()