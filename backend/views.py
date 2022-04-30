from django.http import HttpResponse, JsonResponse

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.exceptions import APIException

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from collections import Counter, OrderedDict

from .utils import classify, preprocess_text, remove_stopwords, createSession, getEncoding, preprocessData
from .logger import logger

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
			df = preprocessData(df)

			# save to csv
			df.to_csv(path_or_buf=f'storage/uploads/{session_id}.csv', index=False)

			df = classify(df, filepath=file_path)

			# save to csv
			df.to_csv(path_or_buf=f'storage/uploads/{session_id}.csv')

		else:
			return Response(status=401)
	else:
		return Response(status=405)

@api_view(['GET'])
def data_overview(request):

	if 'GET' == request.method:
		if request.COOKIES.get('sessionid'):
			session_id = request.COOKIES.get('sessionid')

			df = pd.read_csv(f'storage/uploads/{session_id}.csv')

			data = {}

			data['Overall'] = df.shape[0]

			for brand in ['Pfizer', 'Sinovac', 'Astrazeneca', 'Moderna']:
				per_brand = df.loc[brand == df['Brand']]
				data[brand] = per_brand.shape[0]

			return Response(data=data)
		else:
			return Response(status=401)
	else:
		return Response(status=405)

@api_view(['GET'])
def sentiment_overview(request):

	if 'GET' == request.method:
		if request.COOKIES.get('sessionid'):

			session_id = request.COOKIES.get('sessionid')

			df = pd.read_csv(f'storage/uploads/{session_id}.csv')

			count_dict = dict(Counter(df['Sentiment']))

			data = {}

			data['Overall'] = {
				'positive': count_dict[1],
				'negative': count_dict[0]
			}

			for brand in ['Pfizer', 'Sinovac', 'Astrazeneca', 'Moderna']:
				per_brand = df.loc[brand == df['Brand']]
				count_dict = dict(Counter(per_brand['Sentiment']))

				data[brand] = {
					'positive': count_dict[1],
					'negative': count_dict[0]
				}

			return Response(data=data)
		else:
			return Response(status=401)

	else:
		return Response(status=405)

@api_view(['GET'])
def sentiment_trend(request):

	if 'GET' == request.method:
		if request.COOKIES.get('sessionid'):
			session_id = request.COOKIES.get('sessionid')

			df = pd.read_csv(f'storage/uploads/{session_id}.csv')

			df['Created-At'] = pd.to_datetime(df['Created-At'], infer_datetime_format=True)
			df['Created-At'] = pd.to_datetime(df['Created-At']).dt.to_period('M')
			df['Created-At'] = df['Created-At'].apply(str)

			data = {}

			labels = df['Created-At'].value_counts()
			labels = labels.keys()

			positive_sentiments = {}
			negative_sentiments = {}

			for label in labels:
				data_frame = df.loc[label == df['Created-At']]

				sentiments = data_frame['Sentiment'].value_counts()

				positive_sentiments[label] = sentiments[1] if 1 in sentiments else 0
				negative_sentiments[label] = sentiments[0] if 0 in sentiments else 0

			sentiments = []
			sentiments.append(OrderedDict(sorted(positive_sentiments.items())))
			sentiments.append(OrderedDict(sorted(negative_sentiments.items())))

			data['Overall'] = sentiments

			for brand in ['Pfizer', 'Sinovac', 'Astrazeneca', 'Moderna']:
				per_brand = df.loc[brand == df['Brand']]

				for label in labels:
					data_frame = per_brand.loc[label == per_brand['Created-At']]

					sentiments = data_frame['Sentiment'].value_counts()

					positive_sentiments[label] = sentiments[1] if 1 in sentiments else 0
					negative_sentiments[label] = sentiments[0] if 0 in sentiments else 0

				sentiments = []
				sentiments.append(OrderedDict(sorted(positive_sentiments.items())))
				sentiments.append(OrderedDict(sorted(negative_sentiments.items())))

				data[brand] = sentiments

			return Response(data=data)
		else:
			return Response(status=401)

	else:
		return Response(status=405)

@api_view(['GET'])
def all_data(request):

	if 'GET' == request.method:
		if request.COOKIES.get('sessionid'):
			session_id = request.COOKIES.get('sessionid')

			result = pd.read_csv(f'storage/uploads/{session_id}.csv')

			result = result[['Created-At', 'Text', 'Brand', 'Sentiment']]

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
			logger.info(data)
			return Response(data=data, headers=headers)
		else:
			return Response(status=401)
	else:
		return Response(status=405)

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

