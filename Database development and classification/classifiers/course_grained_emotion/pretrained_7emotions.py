# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np


def load7EmotionsModel():
    #MODEL = "bdotloh/just-another-emotion-classifier"
    access_token = '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'
    MODEL = "emo-nlp/7-emo"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token = access_token)
    #config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, token = access_token)
    return model,tokenizer

def get_emotion_7(text, model,tokenizer):
    #negative, positive, neural
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    #print(scores)
    return scores

