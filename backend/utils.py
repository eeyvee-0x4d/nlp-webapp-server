import nltk
import stopwordsiso
import re
import sklearn
import pickle
import chardet

from nltk.stem import *
from nltk.corpus import stopwords
from nltk.util import ngrams

from django.contrib.sessions.backends.db import SessionStore
from django.contrib.sessions.models import Session

def preprocess_text(data_frame):

	df_shape = data_frame.shape # (row, column)

	# remove urls, remove special chars, conver to lowercase
	for i in range(df_shape[0]):
	  	string = re.sub(r'http\S+', '', data_frame.at[i, 'Text']).lower() # remove url
	  	string = re.sub(r'[^a-zA-Z0-9 ]', '', string) # remove non alpha numeric characters
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

def classify(data_frame):

	model = pickle.load(open('storage/models/model.pkl', 'rb'))
	vectorizer = pickle.load(open('storage/models/vectorizer.pkl', 'rb'))

	corpus = preprocess_text(data_frame)
	corpus = remove_stopwords(corpus)

	corpus = vectorizer.transform(corpus['Text'])
	predictions = model.predict(corpus)

	predictions = pd.Series(predictions)
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

