from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax



def loadSentimentModel():
    #MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    #config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return model,tokenizer

def get_sentiment(text, model,tokenizer):
    #negative, positive, neural
    encoded_input = tokenizer(text, return_tensors='pt', max_length = 514, truncation = True)
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    #print(scores)
    return scores