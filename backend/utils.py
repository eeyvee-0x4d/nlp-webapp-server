import nltk
import stopwordsiso
import re
import sklearn
import pickle
import chardet

import pandas as pd

from nltk.stem import *
from nltk.corpus import stopwords
from nltk.util import ngrams

from django.contrib.sessions.backends.db import SessionStore
from django.contrib.sessions.models import Session

from .logger import logger

def preprocess_text(data_frame):

	df_shape = data_frame.shape # (row, column)

	# remove urls, remove special chars, conver to lowercase
	for i in range(df_shape[0]):
	  	string = re.sub(r'http\S+', '', data_frame.at[i, 'Text']).lower() # remove url
	  	string = re.sub(r'[^a-zA-Z0-9 ]', '', string) # remove non alpha numeric characters
	  	string = re.sub(r'\b\w{1,3}\b', '', string)
	  	data_frame.at[i, 'Text'] = re.sub(r'\n', '', string)

	return data_frame


def remove_stopwords(data_frame):

	stemmer = PorterStemmer() # Porter Stemmer

	stopwords_eng = set(stopwords.words('english')) # English stopwords
	stopwords_tl  = set(stopwordsiso.stopwords('tl'))
	filtered_sentence = []
	filtered_sentence2 = []

	for i in range(len(data_frame['Text'])):
		document = data_frame.loc[i, 'Text']
		tokens = nltk.word_tokenize(document)

		stemmed_tokens = [stemmer.stem(token) for token in tokens] # stem each words
		filtered_sentence = [token for token in stemmed_tokens if not token in stopwords_eng] # remove english stopwords
		filtered_sentence2 = [token for token in filtered_sentence if not token in stopwords_tl] #remove tagalog stopwords

		document = " ".join(filtered_sentence2)
		data_frame.loc[i, 'Text'] = document

	return data_frame

def classify(data_frame, filepath):

	model = pickle.load(open('storage/models/model.pkl', 'rb'))
	vectorizer = pickle.load(open('storage/models/vectorizer.pkl', 'rb'))

	corpus = data_frame
	corpus = preprocess_text(corpus)
	corpus = remove_stopwords(corpus)

	corpus = vectorizer.transform(corpus['Text'])
	# corpus = vectorizer.transform(data_frame['Text'])
	predictions = model.predict(corpus)

	predictions = pd.Series(predictions)

	data_frame = pd.read_csv(filepath)
	data_frame['Sentiments'] = predictions.values

	return data_frame

def createSession(request):

	session = SessionStore()
	session['foo'] = 'bar'
	session.create()

	return session.session_key

def getEncoding(file_path):

	with open(file_path, 'rb') as f:
		result = chardet.detect(f.read(10000))

	return result['encoding']

def preprocessData(data_frame):

	# drop columns except Created-At and Text columns
	data_frame = data_frame[['Created-At', 'Text']]


	# RegEx patterns to match specifi keywords
	PATTERN_1 = r'(RT )'
	PATTERN_2 = re.compile('pfizer|pfizerbiontech|pfizervaccine', re.IGNORECASE)
	PATTERN_3 = re.compile('sinovac|sinovacvaccine|sinovacbiotech', re.IGNORECASE)
	PATTERN_4 = re.compile('astrazeneca|astrazenecavaccine|oxford|oxfordastrazeneca', re.IGNORECASE)
	PATTERN_5 = re.compile('moderna|modernavaccine', re.IGNORECASE)

	for index in range(len(data_frame)):
		if re.match(PATTERN_1, data_frame.at[index, 'Text']):
			data_frame = data_frame.drop([index])
			data_frame.reset_index(drop=True)
		elif re.search(PATTERN_2, data_frame.at[index, 'Text']):
			data_frame.loc[index, 'Brand'] = 'Pfizer'
		elif re.search(PATTERN_3, data_frame.at[index, 'Text']):
			data_frame.loc[index, 'Brand'] = 'Sinovac'
		elif re.search(PATTERN_4, data_frame.at[index, 'Text']):
			data_frame.loc[index, 'Brand'] = 'Astrazeneca'
		elif re.search(PATTERN_5, data_frame.at[index, 'Text']):
			data_frame.loc[index, 'Brand'] = 'Moderna'
		else:
			data_frame = data_frame.drop([index])
			data_frame.reset_index(drop=True)

	# remove duplicates
	data_frame.drop_duplicates(subset=['Text'], inplace=True, ignore_index=True)
	data_frame.reset_index(drop=True)

	return data_frame

