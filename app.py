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
@app.route('/Predict_World_News')
def predict_World_News():
  model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
  obtain_stories = pd.read_csv('World_stories.csv', encoding="UTF-8")
  obtain_comments = pd.read_csv('World_comments.csv', encoding="UTF-8")
  stories = obtain_stories
  obtain_comments.columns = ['Sentence','Label']
  sentences = obtain_comments['Sentence']
  labels = obtain_comments['Label']
  stories_real= stories.values.tolist() 
  story_sentiment =[]
  for j in range(0,5): 
    for i in range (0,19):
        text = sentences[i]
        sequences = tokenizer.texts_to_sequences(text)
        data = pad_sequences(sequences, maxlen=200)
        probability = model.predict(data, batch_size=1, verbose = 2)[0]
        sentiment = np.argmax(probability)  
        labels.loc[i] = sentiment
    n = len(labels) 
    numba = Counter(labels) 
    get_mode = dict(numba) 
    mode = [k for k, v in get_mode.items() if v == max(list(numba.values()))] 
    if len(mode) == n: 
        expression = 'Split Opinion' 
    elif mode [0]==0: 
        expression = 'Sadness'
    elif mode[0]==1:
        expression='Anger'
    elif mode[0]==2:
        expression= 'Neutrality'
    elif mode[0]==3:
        expression ='Happiness' 
    story_sentiment.append(expression)
  table_frame = pd.DataFrame(list(zip(stories_real,story_sentiment)), columns =['News Stories', 'Story Sentiment'])
  table_frame.columns = ['Story','Sentiment']
  HTML_File = table_frame.to_html()
  html_string_1 = '''
<!DOCTYPE HTML>
<!--
	EmoNews by TEMPLATED
	templated.co @templatedco
	Released for free under the Creative Commons Attribution 3.0 license (templated.co/license)
-->
<html>
	<head>
		<title>No Sidebar - Phase Shift by TEMPLATED</title>
		<meta http-equiv="content-type" content="text/html; charset=utf-8" />
		<meta name="description" content="" />
		<meta name="keywords" content="" />
				<!--[if lte IE 8]><script type="text/javascript" src="{{ url_for('static', filename='css/ie/html5shiv.js')}}"></script><![endif]-->
		<script type="text/javascript" src="{{ url_for('static', filename='js/jquery.min.js')}}"></script>
		<script type="text/javascript" src="{{ url_for('static', filename='js/jquery.dropotron.min.js')}}"></script>
		<script type="text/javascript" src="{{ url_for('static', filename='js/skel.min.js')}}"></script>
		<script type="text/javascript" src="{{ url_for('static', filename='js/skel-layers.min.js')}}"></script>
		<script type="text/javascript" src="{{ url_for('static', filename='js/init.js')}}"></script>
		<noscript>
			<link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='css/skel.css')}}"/>
			<link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='css/style.css')}}" />
			<link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='css/style-wide.css')}}" />
		</noscript>
		<!--[if lte IE 8]><link rel="stylesheet" type="text/css" href="{{url_for('static', filename='css/ie/v8.css')}}" /><![endif]-->
        </head>
	<body>

		<!-- Wrapper -->
			<div class="wrapper style1">

				<!-- Header -->
					<div id="header" class="skel-panels-fixed">
						<div id="logo">
							<h1><a href="index.html">EmotioNews</a></h1>
							<span class="tag">by Libazisa Kunda</span>
						</div>
						<nav id="nav">
							<ul>
								<li class="active"><a href="index.html">Homepage</a></li>
							</ul>
						</nav>
					</div>
				<!-- Header -->

				<!-- Page -->
					<div id="page" class="container">
						<section>
							<header class="major">
								<h2>News Stories From Africa</h2>
								<h3 class="byline">These are the news stories making headlines in Africa and the emotions being expressed on those Stories</h3>
							</header>
              <div>
              '''+HTML_File+'''
              </div>
				<!-- /Page -->
				<!-- Copyright -->
		<div id="copyright">
			<div class="container"> <span class="copyright">Design: <a href="http://templated.co">TEMPLATED</a> Images: <a href="http://unsplash.com">Unsplash</a> (<a href="http://unsplash.com/cc0">CC0</a>)</span>
				<ul class="icons">
					<li><a href="#" class="fa fa-facebook"><span>Facebook</span></a></li>
					<li><a href="#" class="fa fa-twitter"><span>Twitter</span></a></li>
					<li><a href="#" class="fa fa-google-plus"><span>Google+</span></a></li>
				</ul>
			</div>
		</div>

	</body>
</html>
'''
  return html_string_1

@app.route('/predict_comment', methods=['POST'])
def predict_comment():
 input_sentence= request.form['message']
 message = [input_sentence]
 sequences = tokenizer.texts_to_sequences(message)
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

@app.route('/Predict_African_Stories')
def Predict_African_Stories():
  model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
  obtain_stories = pd.read_csv('African_stories.csv', encoding="UTF-8")
  obtain_comments = pd.read_csv('African_comments.csv', encoding="UTF-8")
  stories = obtain_stories
  obtain_comments.columns = ['Sentence','Label']
  sentences = obtain_comments['Sentence']
  labels = obtain_comments['Label']
  stories_real= stories.values.tolist() 
  story_sentiment =[]
  for j in range(0,5): 
    for i in range (0,19):
        text = sentences[i]
        sequences = tokenizer.texts_to_sequences(text)
        data = pad_sequences(sequences, maxlen=200)
        probability = model.predict(data, batch_size=1, verbose = 2)[0]
        sentiment = np.argmax(probability)  
        labels.loc[i] = sentiment
    n = len(labels) 
    numba = Counter(labels) 
    get_mode = dict(numba) 
    mode = [k for k, v in get_mode.items() if v == max(list(numba.values()))] 
    if len(mode) == n: 
        expression = 'Split Opinion' 
    elif mode [0]==0: 
        expression = 'Sadness'
    elif mode[0]==1:
        expression='Anger'
    elif mode[0]==2:
        expression= 'Neutrality'
    elif mode[0]==3:
        expression ='Happiness' 
    story_sentiment.append(expression)
  table_frame = pd.DataFrame(list(zip(stories_real,story_sentiment)), columns =['News Stories', 'Story Sentiment'])
  table_frame.columns = ['Story','Sentiment']
  HTML_File = table_frame.to_html()
  html_string_2 = '''
<!DOCTYPE HTML>
<!--
	EmoNews by TEMPLATED
	templated.co @templatedco
	Released for free under the Creative Commons Attribution 3.0 license (templated.co/license)
-->
<html>
	<head>
		<title>No Sidebar - Phase Shift by TEMPLATED</title>
		<meta http-equiv="content-type" content="text/html; charset=utf-8" />
		<meta name="description" content="" />
		<meta name="keywords" content="" />
		<!--[if lte IE 8]><script type="text/javascript" src="{{ url_for('static', filename='css/ie/html5shiv.js')}}"></script><![endif]-->
		<script type="text/javascript" src="{{ url_for('static', filename='js/jquery.min.js')}}"></script>
		<script type="text/javascript" src="{{ url_for('static', filename='js/jquery.dropotron.min.js')}}"></script>
		<script type="text/javascript" src="{{ url_for('static', filename='js/skel.min.js')}}"></script>
		<script type="text/javascript" src="{{ url_for('static', filename='js/skel-layers.min.js')}}"></script>
		<script type="text/javascript" src="{{ url_for('static', filename='js/init.js')}}"></script>
		<noscript>
			<link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='css/skel.css')}}"/>
			<link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='css/style.css')}}" />
			<link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='css/style-wide.css')}}" />
		</noscript>
		<!--[if lte IE 8]><link rel="stylesheet" type="text/css" href="{{url_for('static', filename='css/ie/v8.css')}}" /><![endif]-->
        </head>
	<body>

		<!-- Wrapper -->
			<div class="wrapper style1">

				<!-- Header -->
					<div id="header" class="skel-panels-fixed">
						<div id="logo">
							<h1><a href="index.html">EmotioNews</a></h1>
							<span class="tag">by Libazisa Kunda</span>
						</div>
						<nav id="nav">
							<ul>
								<li class="active"><a href="index.html">Homepage</a></li>
							</ul>
						</nav>
					</div>
				<!-- Header -->

				<!-- Page -->
					<div id="page" class="container">
						<section>
							<header class="major">
								<h2>News Stories From Africa</h2>
								<h3 class="byline">These are the news stories making headlines in Africa and the emotions being expressed on those Stories</h3>
							</header>
              <div>
              '''+HTML_File+'''
              </div>
				<!-- /Page -->
				<!-- Copyright -->
		<div id="copyright">
			<div class="container"> <span class="copyright">Design: <a href="http://templated.co">TEMPLATED</a> Images: <a href="http://unsplash.com">Unsplash</a> (<a href="http://unsplash.com/cc0">CC0</a>)</span>
				<ul class="icons">
					<li><a href="#" class="fa fa-facebook"><span>Facebook</span></a></li>
					<li><a href="#" class="fa fa-twitter"><span>Twitter</span></a></li>
					<li><a href="#" class="fa fa-google-plus"><span>Google+</span></a></li>
				</ul>
			</div>
		</div>

	</body>
</html>
'''
  return html_string_2


if __name__=='__main__':
  app.run(debug=True)
