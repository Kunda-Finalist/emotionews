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

if __name__=='__main__':
  app.run(debug=True)
