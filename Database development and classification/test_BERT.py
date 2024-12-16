from ctypes import Array
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import re
from spellchecker import SpellChecker
import contractions
import CEM as cem
from tqdm import tqdm, trange
from PBC4cip import PBC4cip
from PBC4cip.core.Evaluation import obtainAUCMulticlass
from PBC4cip.core.Helpers import get_col_dist, get_idx_val
import logging
logging.disable(logging.WARNING)
import argparse



def load_exchange_data(df,label_array):
    utterances_1 = df['speaker_utterance'].tolist()
    utterances_2 = df['listener_utterance'].tolist()
    labels = [list(label_array).index(empathy_level) for empathy_level in df['empathy'].tolist()]
    return utterances_1, utterances_2, labels


class TextClassificationDataset(Dataset):
    def __init__(self, first_utterances, second_utterances, labels, tokenizer, max_length):
        self.first_utterances = first_utterances
        self.second_utterances = second_utterances
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.first_utterances)

    def __getitem__(self, idx):
        utterance_1 = self.first_utterances[idx]
        utterance_2 = self.second_utterances[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text = utterance_1, text_pair = utterance_2, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()


def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            #print(outputs)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
            y = pd.DataFrame({'empathy': actual_labels})
            y['empathy'] = y['empathy'] + 1
            #print(y['empathy']+1)
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions),cem.get_cem(predictions,y), predictions




def main(model_pth, num_classes):
    print('Testing BERT model')
    print(model_pth)
    print(f'Num of classes: {num_classes}')


    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_database_dir = '/processed_databases/EmpatheticExchanges/'
    testFile = current_dir + train_database_dir + 'EmpatheticExchanges_test.csv'
    df_test = pd.read_csv(testFile)

    x_test = df_test.drop(columns=['empathy'])
    y_test = df_test['empathy']

    label_array = df_test['empathy'].unique()

    df_test = pd.concat([x_test, y_test], axis=1)
    

    test_utt_1, test_utt_2, test_labels = load_exchange_data(df_test,label_array)

    bert_model_name = 'bert-base-uncased'

    max_length = 120
    batch_size = 16

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier(bert_model_name, num_classes).to(device)
    model.load_state_dict(torch.load(current_dir+'/'+model_pth))
    print(device)

    test_dataset = TextClassificationDataset(test_utt_1, test_utt_2, test_labels, tokenizer, max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    accuracy, report, cem, test_predictions = evaluate(model, test_dataloader, device)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(report)
    print(f"Closeness Evaluation Measure: {cem:.4f}")

    with open(current_dir +'/BERT_results/' + "BERT_results_"+str(model_pth)[:-4]+".txt", "w") as f:
        print('Metrics', file = f)
        print(f"\n\nacc: {accuracy}, cem: {cem}", file=f)

    with open(current_dir +'/BERT_results/' + "BERT_predictions_"+str(model_pth)[:-4]+".txt", "w") as f:
        for prediction in test_predictions:
            print(f"{prediction}",file=f)





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test a BERT-based classification algorithm')
    parser.add_argument('--model', metavar='model', required=True,
                        help="State dictionary of the model to test (including the .pth). Example = bert_classifier_3_levels.pth")
    parser.add_argument('--num_of_classes', metavar='model', required=True,
                        help='number of class labels')
    args = parser.parse_args()
    main(str(args.model), int(args.num_of_classes))