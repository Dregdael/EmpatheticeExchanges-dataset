import pickle
import pandas as pd
import torch
import os

from PBC4cip import PBC4cip
import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm, trange
from PBC4cip import PBC4cip
from PBC4cip.core.Evaluation import obtainAUCMulticlass
from PBC4cip.core.Helpers import get_col_dist, get_idx_val



def train(experiment_number,data_train):
    current_dir = os.path.dirname(os.path.abspath(__file__))


    pbc = PBC4cip(tree_count = 150,filtering=False, multivariate = False) # create classifier with custom tree count
 
 
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

