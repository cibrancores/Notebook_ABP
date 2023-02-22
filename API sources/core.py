import pandas
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from fastapi import FastAPI
from fastapi import Response

SEARCHTITLE = "default"

api = FastAPI()

@api.on_event("startup")
async def start_api():
	global data
	data = loadDataset()
	print("=============================")
	print("=== MÅCIAGGOJÖ IS RUNNING ===")
	print("=============================")

@api.get("/")
async def root():
	return "[!] Welcome to MÅCIAGGOJÖ, the CRÅGYE API. Chech out /docs for details."

@api.get("/rcm")
async def rcm_manager(t: str):
	return recommend(t, data)

@api.get("/item")
async def item_manager(id: str):
	return get_item(id, data)

def loadDataset():
	nltk.download('punkt')
	nltk.download('stopwords')
	dataset = pandas.read_csv("src.csv", dtype={'id': int})
	dataset.fillna("", inplace=True)
	return dataset

def recommend(descrip, data):
	newRow = {'id': 0, 'name': SEARCHTITLE, 'price': 0, 'description': descrip}
	data = data.append(newRow, ignore_index=True)

	ps = PorterStemmer()
	newPreprocessedText = []
	for row in data.itertuples():
		text = word_tokenize(row[5])
		stops = set(stopwords.words("english"))
		text = [ps.stem(w) for w in text if w not in stops and w.isalnum()]
		text = " ".join(text)

		newPreprocessedText.append(text)

	newPreprocessedData = data
	newPreprocessedData['processed_text'] = newPreprocessedText

	bagOfWordsModel = TfidfVectorizer()
	bagOfWordsModel.fit(newPreprocessedData['processed_text'])
	textsBoW = bagOfWordsModel.transform(newPreprocessedData['processed_text'])

	distance_matrix = pairwise_distances(textsBoW, textsBoW, metric='cosine')

	indexOfTitle = newPreprocessedData[newPreprocessedData['name'] == SEARCHTITLE].index.values[0]
	distance_scores = list(enumerate(distance_matrix[indexOfTitle]))
	ordered_scores = sorted(distance_scores, key=lambda x: x[1])
	top_scores = ordered_scores[1:11]
	top_indexes = [i[0] for i in top_scores]
	newPreprocessedData.drop(newPreprocessedData.tail(1).index, inplace=True)

	del newPreprocessedData['processed_text']

	return newPreprocessedData.iloc[top_indexes].to_dict("records")

def get_item(id, data):
	return data[data['id'] == int(id)].to_dict("records")
