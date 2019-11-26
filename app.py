from collections import Counter 
from flask import Flask, render_template,url_for,request
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

model = load_model('EmotioNewsV2.h5')
with open('tokenizer2.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/comment')
def comment():
    return render_template('comment.html')

@app.route('/Predict_African_Stories')
def Predict_African_Stories():
 return render_template('Africa.html')

@app.route('/Predict_World_Stories')
def predict_World_News():
  return render_template('World.html')

@app.route('/predict_comment', methods=["POST","GET"])
def predict_comment():
 if request.method == "POST":
  input_sentence= request.form["message"]
  message_text=[input_sentence]
  sequences = tokenizer.texts_to_sequences(message_text)
  data = pad_sequences(sequences, maxlen=200)
  #Exporting and Loading of the Deep Learning Model
  model.compile(loss='sparse_categorical_crossentropy',
        	      optimizer='adam',
               	      metrics=['accuracy'])
  sentiment = model.predict(data, batch_size=1, verbose = 2)[0]
  if(np.argmax(sentiment) == 0):
   message_prediction = 0
  elif (np.argmax(sentiment) == 1):
   message_prediction = 1
  elif (np.argmax(sentiment) == 2):
   message_prediction = 2
  elif (np.argmax(sentiment) == 3):
   message_prediction = 3
 return render_template('emotion.html', prediction = message_prediction) 

if __name__=='__main__':
  app.run(debug=True)
