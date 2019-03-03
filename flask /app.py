import flask
import numpy as np
import pandas as pd
import pickle
import itertools
import json
import seaborn as sns
import math
import nltk, string
import re
import random
import nltk, string,re
from nltk.corpus import stopwords
from textblob import TextBlob
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim.corpora import Dictionary


print(simple_preprocess('hello please parse this'))

with open('car_df_final', 'rb') as picklefile:
	car_df_final = pickle.load(picklefile)

with open('lda_model', 'rb') as picklefile:
	lda_model = pickle.load(picklefile)

with open('car_df_cleaned', 'rb') as f:
    car_df = pickle.load(f)




app = flask.Flask(__name__)

@app.route("/")
def viz_page():
    with open("car_app.html", 'r') as viz_file:
        return viz_file.read()








@app.route("/cars", methods=["POST"])
def answer():

    def lemmatize_stemming(text):
        return WordNetLemmatizer().lemmatize(text, pos='v')
    def preprocess(text):
        result = []
        for token in simple_preprocess(text):
            if token not in stop and len(token) > 2:
                result.append(lemmatize_stemming(token))
        return result

    def get_top_cars(new_car_df,topicnum,num_cars):
        return new_car_df[new_car_df.topic_num == topicnum].sort_values(['topicprob'],ascending=False)[:num_cars]

    def get_top_cars_filter(new_car_df,topicnum,num_cars,carfilter):
        return new_car_df[(new_car_df.topic_num == topicnum) & (new_car_df.type == carfilter)].sort_values(['topicprob'],ascending=False)[:num_cars]

    stop = stopwords.words('english');
    stop.append('want')
    stop.append('car')
    stop.append('like')



    data = flask.request.json
    test_case_list = data["question"]
    test = test_case_list[0]


    #Create a dictionary from ‘processed_docs’ containing the number of times a word appears in the training set.
    processed_docs = car_df.review.str.split()
    dictionary = Dictionary(processed_docs)
    #tmp = re.sub(r'\b\w{1,2}\b', '', test)
    tmp2 = preprocess(test)
    bow_vector = dictionary.doc2bow(tmp2)
    topic_num = sorted(lda_model[bow_vector], key=lambda x:x[1],reverse=True)[0][0]

    automobile_types =['truck','suv','convertible','sedan','hatchback','van','coupe']
    word_tmp = []
    for word in tmp2:
        if word in automobile_types:
            word_tmp.append(word)

    if len(word_tmp) == 0:
        new = get_top_cars(car_df_final,topic_num,10).vehicle.values
    else:   
        new = get_top_cars_filter(car_df_final,topic_num,10,word_tmp[0]).vehicle.values

    answer = list(new)

    
    the_answer = answer
    return flask.jsonify({'answer':the_answer})

app.run(debug=True, threaded=True)