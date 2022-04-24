from django.http import HttpResponse, JsonResponse

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.exceptions import APIException

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from collections import Counter, OrderedDict

from .utils import preprocess_text, remove_stopwords, createSession, getEncoding

import codecs
import sklearn
import pickle
import re
import os
import traceback
import json
import chardet

import pandas as pd


"""
This end point will create a session cookie and send back the id
of the session
"""
@api_view(['GET'])
def session(request):
	
	if request.query_params['get_cookie']:
		session_key = createSession(request)

		data = {
			'sessionid': session_key
		}

		response = Response(data=data)
		return response
	else:
		return Response(status=301)

"""
This endpoint will save the file uploaded and preprocess the file.
"""
@api_view(['POST'])
def upload(request):

	if request.method == 'POST':
		if request.COOKIES.get('sessionid'):
			session_id = request.COOKIES.get('sessionid')

			uploaded_file = request.FILES['file']

			file_path = f"storage/uploads/{session_id}.csv"

			if os.path.exists(file_path):
				os.remove(file_path)
				file_path = default_storage.save(f"storage/uploads/{session_id}.csv", ContentFile(uploaded_file.read()))

			else:
				file_path = default_storage.save(f"storage/uploads/{session_id}.csv", ContentFile(uploaded_file.read()))

			file_encoding = getEncoding(file_path)
			
			df = pd.read_csv(file_path, encoding=file_encoding)
			
			# TODO: text preprocessing
		else:
			return Response(status=301)

	response = Response()

	return response

@api_view(['GET'])
def data_overview(request):

	data = {
		'pfizer':  None,
		'sinovac':  None,
		'astrazeneca':  None,
		'moderna':  None,
	}

	for key in data:
		try:
			df = pd.read_csv('storage/uploads/' + key + '/' + key + '.csv')
			num_tweets = df.shape[0]
			data[key] = num_tweets
		except:
			data[key] = None

	return Response(data)

@api_view(['GET'])
def sentiment_overview(request):

	data = [
		{
			'name': 'pfizer',
			'sentiments': [
				{
					'positive': 0,
					'negative': 0
				}
			]
		},
		{
			'name': 'sinovac',
			'sentiments': [
				{
					'positive': 0,
					'negative': 0
				}
			]
		},
		{
			'name': 'astrazeneca',
			'sentiments': [
				{
					'positive': 0,
					'negative': 0
				}
			]
		},
		{
			'name': 'moderna',
			'sentiments': [
				{
					'positive': 0,
					'negative': 0
				}
			]
		},
	]

	for item in data:
		try:
			df = pd.read_csv('storage/uploads/' + item['name'] + '/_' + item['name'] + '.csv')

			count_dict = dict(Counter(df['Sentiment']))
			item['sentiments'] = [
				{
					'positive': count_dict[1],
					'negative': count_dict[0]
				}
			]

		except FileNotFoundError:
			item['sentiments'] = None
		except Exception as e:
			raise APIException(repr(e))

	return Response(data)

@api_view(['GET'])
def sentiment_trend(request):

	# model = pickle.load(open('storage/models/model.pkl', 'rb'))
	# vectorizer = pickle.load(open('storage/models/vectorizer.pkl', 'rb'))

	data = [
		{
			'name': 'pfizer',
			'sentiments': []
		},
		{
			'name': 'sinovac',
			'sentiments': []
		},
		{
			'name': 'astrazeneca',
			'sentiments': []
		},
		{
			'name': 'moderna',
			'sentiments': []
		},
	]

	for item in data:
		try:
			df = pd.read_csv('storage/uploads/' + item['name'] + '/_' + item['name'] + '.csv')

			df_shape = df.shape
			for i in range(df_shape[0]):
				s = re.sub(r' (\d{2}):(\d{2}):(\d{2})', '', df.at[i, 'Created-At']).split('-')
				s = s[0] + '/' + s[1]
				df.at[i, 'Created-At'] = s
			
			# df = preprocess_text(df)
			# df = remove_stopwords(df)

			# corpus = vectorizer.transform(df['Text'])
			# predictions = model.predict(corpus)

			# predictions = pd.Series(predictions)
			# df['Sentiments'] = predictions.values

			labels = df['Created-At'].value_counts()
			labels = labels.keys()

			positive_sentiments = {}
			negative_sentiments = {}

			for label in labels:
				df_shape = df.shape
				pos_sentiments = 0
				neg_sentiments = 0
				for i in range(0, df_shape[0]):
					if(df.at[i, 'Created-At'] == label and df.at[i, 'Sentiment'] == 1):
					 	pos_sentiments += 1
					elif(df.at[i, 'Created-At'] == label and df.at[i, 'Sentiment'] == 0):
					  	neg_sentiments += 1

				positive_sentiments[label] = pos_sentiments
				negative_sentiments[label] = neg_sentiments

			item['sentiments'].append(OrderedDict(sorted(positive_sentiments.items())))
			item['sentiments'].append(OrderedDict(sorted(negative_sentiments.items())))

		except FileNotFoundError:
			item['sentiments'] = None
		except Exception as e:
			raise APIException(repr(e))

	return Response(data)

@api_view(['GET'])
def all_data(request):

	frames = []

	if os.path.exists('storage/uploads/pfizer/_pfizer.csv'):
			frame_1 = pd.read_csv('storage/uploads/pfizer/_pfizer.csv')
			frame_1 = frame_1[['Created-At','Text', 'Sentiment']]
			frames.append(frame_1)

	if os.path.exists('storage/uploads/sinovac/_sinovac.csv'):
			frame_2 = pd.read_csv('storage/uploads/sinovac/_sinovac.csv')
			frame_2 = frame_2[['Created-At','Text', 'Sentiment']]
			frames.append(frame_2)

	if os.path.exists('storage/uploads/astrazeneca/_astrazeneca.csv'):
			frame_3 = pd.read_csv('storage/uploads/astrazeneca/_astrazeneca.csv')
			frame_3 = frame_3[['Created-At','Text', 'Sentiment']]
			frames.append(frame_3)

	if os.path.exists('storage/uploads/moderna/_moderna.csv'):
			frame_4 = pd.read_csv('storage/uploads/moderna/_moderna.csv')
			frame_4 = frame_4[['Created-At','Text', 'Sentiment']]
			frames.append(frame_4)

	result = pd.concat(frames, ignore_index=True, sort=False)

	rows = result.shape[0]
	# return Response(result.to_json(orient='index'))
	page_number = 0 if (request.GET.get('page_number') is None) else int(request.GET.get('page_number')) - 1
	page_size = 20 if (request.GET.get('page_size') is None) else int(request.GET.get('page_size'))

	start = page_number * page_size
	end = (start + page_size) if ((start + page_size) < rows) else rows
	# print(result.iloc[start:end])

	headers = {}
	headers['Total-Count'] = rows
	headers['Total-Pages'] = int(round(rows / page_size, 0))
	headers['Current-Page'] = page_number + 1
	headers['Page-Size'] = page_size
	# print(headers)

	data = result.iloc[start:end]
	data = data.to_json(orient='index')

	# print(result.iloc[0:10])
	return Response(data=data, headers=headers)

@api_view(['GET'])
def model_stats(request):

	data = ''

	with open('storage/models/model_stats.json') as f:
		data = json.load(f)

	return Response(data)

@api_view(['GET'])
def test(request):
	print(request.session.decode())
	return Response("Hello")

