import pickle
import pandas as pd
import torch
import os

import CEM as cem

from PBC4cip import PBC4cip
import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm, trange
from PBC4cip import PBC4cip
from PBC4cip.core.Evaluation import obtainAUCMulticlass
from PBC4cip.core.Helpers import get_col_dist, get_idx_val

def score(predicted, y):
        y_class_dist = get_col_dist(y[f'{y.columns[0]}'])
        real = list(map(lambda instance: get_idx_val(y_class_dist, instance), y[f'{y.columns[0]}']))
        numClasses = len(y_class_dist)
        confusion = [[0]* numClasses for i in range(numClasses)]
        classified_as = 0
        error_count = 0

        for i in range(len(real)):
            if real[i] != predicted[i]:
                error_count = error_count + 1
            confusion[real[i]][predicted[i]] = confusion[real[i]][predicted[i]] + 1

        acc = 100.0 * (len(real) - error_count) / len(real)
        auc = obtainAUCMulticlass(confusion, numClasses)
        return confusion, acc, auc


def test_single_instance(model_path, data_test):
    pbc = pickle.load(open(model_path, 'rb'))
    data_test["empathy"] = data_test["empathy"].astype('int')
    data_test["empathy"] = data_test["empathy"].astype('string')
    x_test = data_test.drop(columns=['empathy'])
    y_test = data_test.copy()
    y_test = y_test.drop(columns=x_test.columns)
    y_pred = pbc.predict(x_test)
    for prediction in y_pred:
        print(f"{prediction}")
    return 0




def test(experiment_number,data_test,model_path,features_used):

    #setup of directories 
    current_dir = os.path.dirname(os.path.abspath(__file__))

    #get test file to compare
    #testFile = current_dir + current_db + 'test.csv'
    #data_test = pd.read_csv(testFile)
    data_test["empathy"] = data_test["empathy"].astype('int')
    data_test["empathy"] = data_test["empathy"].astype('string')

    x_test = data_test.drop(columns=['empathy'])
    y_test = data_test.copy()
    y_test = y_test.drop(columns=x_test.columns)

    #get model
    pbc = pickle.load(open(model_path, 'rb'))

    #predict with model
    y_pred = pbc.predict(x_test)

    #get evaluation metric
    confusion, acc, auc = score(y_pred, y_test)
    ClosenessEvaluationMeasure = cem.get_cem(y_pred,y_test)
    
    #send out predictions
    with open(current_dir + '/Experiments/outputs/Experiment '+ str(experiment_number) + '/' + "predictions.txt", "w") as f:
        for prediction in y_pred:
            print(f"{prediction}",file=f)

    #print metrics
    print(f"\nConfusion Matrix:")
    print(f'row: true, column: predicted ')
    for i in range(len(confusion[0])):
        for j in range(len(confusion[0])):
            print(f"{confusion[i][j]} ", end='')
        print("")
    print(f"\n\nacc: {acc} , auc: {auc}, cem: {ClosenessEvaluationMeasure}")
    
    #Send results to output folder 
    with open(current_dir + '/Experiments/outputs/Experiment '+ str(experiment_number) + '/' + "results.txt", "w") as f:
        print(features_used, file=f)
        print('', file=f)
        print(f"Confusion Matrix:", file=f)
        print(f'row: true, column: predicted ', file=f)
        for i in range(len(confusion[0])):
            for j in range(len(confusion[0])):
               print(f"{confusion[i][j]} ", end='', file=f)
            print("", file=f)
        print(f"\n\nacc: {acc} , auc: {auc}, cem: {ClosenessEvaluationMeasure}", file=f)
        
    
