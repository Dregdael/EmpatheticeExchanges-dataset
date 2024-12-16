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

def train(experiment_number,data_train):
    current_dir = os.path.dirname(os.path.abspath(__file__))


    pbc = PBC4cip(tree_count = 100,filtering=False, multivariate = False) # create classifier with custom tree count
 
 
    #data_train = pd.read_csv(trainFile)
    data_train["empathy"] = data_train["empathy"].astype('int')
    data_train["empathy"] = data_train["empathy"].astype('string')
    #data_train["speaker_emotion"] = data_train["speaker_emotion"].astype('category')
    #data_train["listener_emotion"] = data_train["listener_emotion"].astype('category')

    x_train = data_train.drop(columns=['empathy'])
    #print(x_train)
    y_train = data_train.copy()
    y_train = y_train.drop(columns=x_train.columns)
    #print(y_train)
    
    #training
    patterns = pbc.fit(x_train,y_train)
    filepath = current_dir + '/Experiments/outputs/Experiment ' + str(experiment_number) + '/'
    filename = 'trained_pbc4cip.sav'

    pickle.dump(pbc, open(filepath + filename, 'wb'))

    with open(filepath + "patterns.txt", "w") as f:
        for pattern in patterns:
            print(f"{pattern}",file=f)

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
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
            y = pd.DataFrame({'empathy': actual_labels})
            y['empathy'] = y['empathy'] + 1
            #print(y['empathy']+1)
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions),cem.get_cem(predictions,y), predictions




def main():

    print('Training BERT model')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_database_dir = '/processed_databases/EmpatheticExchanges/'
    trainFile = current_dir + train_database_dir + 'EmpatheticExchanges_train_binary.csv'
    testFile = current_dir + train_database_dir + 'EmpatheticExchanges_test_binary.csv'
    df_train = pd.read_csv(trainFile)
    df_test = pd.read_csv(testFile)
    #print(df_train.head())
    #print(df_test.head())
    x = df_train.drop(columns=['empathy'])
    y = df_train['empathy']
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)
    x_test = df_test.drop(columns=['empathy'])
    y_test = df_test['empathy']

    label_array = df_test['empathy'].unique()
    #print(label_array)

    df_train = pd.concat([x_train, y_train], axis=1)
    df_test = pd.concat([x_test, y_test], axis=1)
    df_valid = pd.concat([x_valid, y_valid], axis=1)
    

    train_utt_1, train_utt_2, train_labels = load_exchange_data(df_train,label_array)
    test_utt_1, test_utt_2, test_labels = load_exchange_data(df_test,label_array)
    valid_utt_1, valid_utt_2, valid_labels = load_exchange_data(df_valid,label_array)

    bert_model_name = 'bert-base-uncased'
    num_classes = 2
    max_length = 120
    batch_size = 16
    num_epochs = 150
    learning_rate = 2e-5

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    train_dataset = TextClassificationDataset(train_utt_1,train_utt_2, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(valid_utt_1, valid_utt_2, valid_labels, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier(bert_model_name, num_classes).to(device)

    print(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_dataloader, optimizer, scheduler, device) 
        accuracy, report, cem,_ = evaluate(model, val_dataloader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(report)
        print(f"Closeness Evaluation Measure: {cem:.4f}")
        torch.save(model.state_dict(), "bert_classifier_2_extendeddatabase.pth")

    torch.save(model.state_dict(), "bert_classifier_2_extendeddatabase.pth")
    test_dataset = TextClassificationDataset(test_utt_1, test_utt_2, test_labels, tokenizer, max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    accuracy, report, cem, test_predictions = evaluate(model, test_dataloader, device)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(report)
    print(f"Closeness Evaluation Measure: {cem:.4f}")



    with open(current_dir + "BERT_results_2_extended.txt", "w") as f:
        print('Predictions', file = f)
        for prediction in test_predictions:
            print(f"{prediction}",file=f)
        print('Metrics', file = f)
        print(f"\n\nacc: {accuracy}, cem: {cem}", file=f)


if __name__ == '__main__':

    main()