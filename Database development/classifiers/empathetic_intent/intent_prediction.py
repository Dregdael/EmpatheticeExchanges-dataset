import os
from turtle import mode
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import AdamW
from torch import cuda
import pandas as pd


def get_empathetic_intent(text, model, tokenizer,device):
    #pass text to the tokenizer
    encodings = tokenizer(text)
    #turn encodings to tensors and send to device
    encode2Torch = {key: torch.tensor(val).unsqueeze(0).to(device) for key, val in encodings.items()}
    #evaluation mode with no gradient 
    model.eval()
    with torch.no_grad():
        #pass to model
        output = model(**encode2Torch)
        #get prediction from logits
        preds = output.logits.argmax(dim=-1)
        #return prediction
        return preds.item()
    

def get_empathetic_intent_logits(text, model, tokenizer,device):
    #pass text to the tokenizer
    encodings = tokenizer(text)
    #turn encodings to tensors and send to device
    encode2Torch = {key: torch.tensor(val).unsqueeze(0).to(device) for key, val in encodings.items()}
    #evaluation mode with no gradient 
    model.eval()
    with torch.no_grad():
        #pass to model
        output = model(**encode2Torch)
        #get prediction from logits
        preds = output.logits.softmax(dim=-1)
        probs = preds[0].tolist()
        #preds.to('gpu')
        #pred = preds.argmax(dim = -1)
        #return prediction
        return probs

def loadModelTokenizerAndDevice(modelDir):
    #get device
    device = 'cuda' if cuda.is_available() else 'cpu'
    #Create same model as the one used in training 
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.num_labels = 9
    #load model
    model = BertForSequenceClassification(config)
    model.load_state_dict(torch.load(str(modelDir)+'./best_model_sd.pt'))
    model.to(device)
    #Create tokenizer to process text 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model,tokenizer,device

'''
def main():


    model,tokenizer,device = loadModelTokenizerAndDevice()
    
    test_df = pd.DataFrame({'text': ['You sound prepared for it! Im sure you have practiced.','Oh my gosh, Im so sorry that happened.']})

    print(test_df)



    #text to process
    #text = 'You sound prepared for it! Im sure you have practiced.'
    #print(f'text: {text}')

    #get prediction
    #print(get_empathetic_intent(text,model,tokenizer,device))

    test_df['intents'] = test_df['text'].apply(get_empathetic_intent,args =(model,tokenizer,device))
    print(test_df)


if __name__ == '__main__':

    main()
'''