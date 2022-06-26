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
from urllib.request import urlopen, Request
from transformers import AutoConfig

url = "https://sjquillen.medium.com/is-learning-a-language-a-waste-of-time-5d7d8cde57e8"
hdr = {'User-Agent': 'Mozilla/5.0'}

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def GetFakeOrNot(text):
    example = PrepareExample(text)
    out = round(torch.nn.Sigmoid()(model(example)).item())
    if out == 0:
        return "This article has fake elements. It may not be a reliable source of information"
    else:
        return "This article successfully passes our Fake News detection. It is most likely a reliable source of information."

class cfg():
    max_len = 512
    model_name = "microsoft/deberta-v3-base"
    train_batch_size = 2
    valid_batch_size = 16
    fold = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bilstm_hidden = 256
    epochs = 1
    n_folds = 5
    debug = False
    train_folds = [0]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
class WeightedLayerPooling(torch.nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else torch.nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average

class FeedBackModel(torch.nn.Module):
    def __init__(self):
        super(FeedBackModel, self).__init__()
        tconfig = AutoConfig.from_pretrained(cfg.model_name)
        tconfig.update({'output_hidden_states':True})
        self.model = AutoModel.from_pretrained(cfg.model_name, config=tconfig)
        self.model.base_model.embeddings.requires_grad_(False)
        self.fc = torch.nn.Linear(tconfig.hidden_size, 1)
        self.pooler = WeightedLayerPooling(tconfig.num_hidden_layers, layer_start=9, layer_weights=None)
        self.fc_dp = torch.nn.Dropout(0.2)

    def forward(self, inputs):
        out_e = self.model(**inputs)
        out = torch.stack(out_e["hidden_states"])
        out = self.pooler(out)
        outputs = self.fc(self.fc_dp(out[:, 0]))
        return outputs
model = FeedBackModel()
model.load_state_dict(torch.load("SleekApp/deberta_base_epoch_1_fold_1.pth", map_location="cpu"))
def PrepareExample(text):
    inputs = cfg.tokenizer(text, truncation=True, max_length=512, padding="max_length")
    inputs = {k : torch.tensor(v).unsqueeze(0) for (k, v) in inputs.items()}
    return inputs

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import tokenize
nlp = spacy.load("en_core_web_sm")
def finalKeyWords(url, num_display):
    text = getText(url, False)
    return_content = [GetEntities(text), SentenceWordImportance(text, num_display)]
    for t in return_content[1]:
        print(t + "\n")
    return return_content
# give parameter of True if isMedium Article
def getText(link, isMedium):
    bodyText = ""
    req = Request(link,headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page, features="html5lib")
    if isMedium:
        textBoxes = list(soup.find("article").find_all("p"))
    else:
        textBoxes = list(soup.find_all("p"))
    for t in textBoxes:
        bodyText = bodyText + t.get_text() + '. '
    return bodyText

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
def GetMostSimilar(text):
    example = PrepareExample(text)
    example = {k : v for (k, v) in example.items()}
    with torch.no_grad():
        text_embed = model(**example)[1].squeeze().detach().cpu().numpy()
    similarity_scores = np.zeros(len(embeddings))
    for i in range(len(embeddings)):
        similarity_scores[i] = np.corrcoef(text_embed, embeddings[i])[0][1]
    return data["Content"].values[np.argmax(similarity_scores)]
def GetSummary(text):
    return summarizer(" ".join(data["Content"].values[2].split()[:800]))[0]["summary_text"]
@app.route("/")
@app.route("/home")
def home():
	return render_template("index.html")

@app.route("/result",methods = ['POST'])
def result():
	content = request.form.to_dict()
	name1 = content["name"]
	return render_template("index.html",inp = [SentenceWordImportance(getText(name1,True),3),getText(name1, True),GetFakeOrNot(getText(name1,True))])

if __name__ == '__main__':
	app.run(debug = True,port = 5001)