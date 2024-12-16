# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

tokenizer = AutoTokenizer.from_pretrained("bdotloh/distilbert-base-uncased-go-emotion-empathetic-dialogues-context-v2")



def load32EmotionsModel():
    #MODEL = "bdotloh/just-another-emotion-classifier"
    MODEL = "bdotloh/distilbert-base-uncased-go-emotion-empathetic-dialogues-context-v2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    #config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return model,tokenizer

def get_emotion_32(text, model,tokenizer):
    #negative, positive, neural
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    #print(scores)
    return scores