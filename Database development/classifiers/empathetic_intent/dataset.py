from turtle import shape
from unittest.util import _MAX_LENGTH
import torch
from transformers import BertTokenizer
import pandas as pd

"""
    Preprocess the Empathetic Intents dataset
    -Obtain data from text files
    -turn it into dataframes
    -process it using the tokenizer 
    -turn it into iterable datasets using train, test, and valid as keys
"""


class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, frameofdata,tokenizer):
        self.dataframe = frameofdata
        self.raw_text = frameofdata.text
        self.labels = frameofdata.label
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        #item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        text2Proccess = str(self.raw_text[idx])
        label2Return = (self.labels[idx])
        tokenizerOut = self.tokenizer(text2Proccess,padding = 'max_length', truncation = True, max_length = 200, pad_to_max_length = True)
        ids = tokenizerOut['input_ids']
        #print(len(ids))
        mask = tokenizerOut['attention_mask']
        #print(len(mask))
        token_type_ids = tokenizerOut['token_type_ids']


        return {
            'input_ids': torch.tensor(ids),
            'token_type_ids' : torch.tensor(token_type_ids),
            'attention_mask': torch.tensor(mask),
            'labels': torch.tensor(label2Return) 
        }

    def __len__(self):
        return len(self.labels)

def get_data(path):

    dataset = {}
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #print('get data')
    #print(tokenizer)

    df_dict = {} #dataframe of dictionaries containing each the train, valid, and test dataframes

    #for each type of dataset
    for taipu in ['train', 'valid', 'test']:
        
        #open text file, get text and labels and place them in a list
        data_path = path + '/' + taipu + '.txt'
        text = []
        label = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                label.append(int(line[0]))
                text.append(line[1])
        #get lists and turn them into a dictionary
        dictio = {'text': text, 'label': label}
        #turn dictionary into a dataframe and place them in a dictionary
        df_dict[taipu] = pd.DataFrame(dictio)
        #print(f'Longest string in {taipu}: {df_dict[taipu]['text'].str.len().max()}')
        #get the Dataset type version of the dataframe, each entry returns ids, an attention mask, token_type_ids, and labels.
        #All of these returns are already pytorch tensors
        dataset[taipu] = IntentDataset(df_dict[taipu],tokenizer)

    #return all three datasets, this is a dictionary, so each can be accessed using the train, valid, and test keys
    return dataset
