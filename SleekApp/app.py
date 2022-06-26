from flask import Flask,render_template,request,Response
import requests
app = Flask(__name__)
from transformers import Trainer, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

import numpy as np
import os
import numpy as np
from flask import Response
import spacy
import nltk
from bs4 import BeautifulSoup
import torch
from collections import Counter
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "true"
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
model = AutoModel.from_pretrained("SleekApp/checkpoint-7800")
data = pd.read_csv("SleekApp/dataset_first_100000.csv")
embeddings = np.load("SleekApp/news_embeddings (1).npy")
def PrepareExample(text):
    inputs = tokenizer(text, truncation=True, max_length=512, padding="max_length")
    inputs = {k : torch.tensor(v).unsqueeze(0) for (k, v) in inputs.items()}
    return inputs

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import tokenize
nlp = spacy.load("en_core_web_sm")
def finalKeyWords(url, num_display):
    text = getText(url)
    return_content = [GetEntities(text), SentenceWordImportance(text, num_display)]
    for t in return_content[1]:
        print(t + "\n")
    return return_content
def getText(url):
	page = requests.get(url)
	soup = BeautifulSoup(page.content, "html.parser")
	global text 
	text = soup.get_text()
	return text

def GetEntities(text):
    doc = nlp(text)
    items = [e.text for e in doc.ents if e.label_ == "ORG"]
    items = [(x, y) for (x, y) in dict(Counter(items)).items()]
    return items
def SentenceWordImportance(text, num_display):
    sentences = tokenize.sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    words_tfidf = vectorizer.fit_transform(sentences)
    important_sentences = []
    sent_sum = words_tfidf.sum(axis=1)
    important_sent = np.argsort(sent_sum, axis=0)[::-1]

    for i in range(0, len(sentences)):
        if i in important_sent[:num_display]:
            important_sentences.append(sentences[i])
    return important_sentences
ur = "https://devonprice.medium.com/thoughts-are-not-feelings-is-shitty-psychological-advice-b5a292d0a7f1"
# print(finalKeyWords(ur, 3)[1])
# print(GetEntities(getText(ur)))
def GetMostSimilar(url):
    text = getText(url)
    example = PrepareExample(text)
    example = {k : v for (k, v) in example.items()}
    with torch.no_grad():
        text_embed = model(**example)[1].squeeze().detach().cpu().numpy()
    similarity_scores = np.zeros(len(embeddings))
    for i in range(len(embeddings)):
        similarity_scores[i] = np.corrcoef(text_embed, embeddings[i])[0][1]
    return data["Content"].values[np.argmax(similarity_scores)]
@app.route("/")
@app.route("/home")
def home():
	return render_template("index.html")

@app.route("/result",methods = ['POST'])
def result():
	content = request.form.to_dict()
	name1 = content["name"]
	return render_template("index.html",inp = [SentenceWordImportance(getText(name1),3),0,GetMostSimilar(name1)])

if __name__ == '__main__':
	app.run(debug = True,port = 5001)
