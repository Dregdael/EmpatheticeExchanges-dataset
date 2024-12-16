import pickle
import pandas as pd
import torch
import os
import sys

from PBC4cip import PBC4cip
import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm, trange
from PBC4cip import PBC4cip
from PBC4cip.core.Evaluation import obtainAUCMulticlass
from PBC4cip.core.Helpers import get_col_dist, get_idx_val
import train_classifier as trainer
import test_classifier as tester
import database_processing_package as data_processer

#for iteration for experiments
import itertools


current_dir = os.path.dirname(os.path.abspath(__file__))
#Available databases
database_dir_ec = '/processed_databases/EmpatheticConversationsExchangeFormat/'
database_dir_ex = '/processed_databases/EmpatheticExchanges/'

#Experiment parameters
experiment_number = 114
#whether to do training or use an already trained model
do_training = 1
#choose training database
train_database_dir = database_dir_ex
#choose testing database
test_database_dir = database_dir_ex
#already trained model
already_trained_model_path = current_dir + '/Experiments/outputs/Experiment '+ str(113) + '/' + 'trained_pbc4cip.sav'
#whether to reprocess the database
reprocess_database = 1
#automated processing flag 
auto_experiments = 0
#control vector for database processing
database_control_vector = [ 1,#database to classify 0 = empatheticconversations (old), 1 empatheticexchanges (new), selected automatically when reprocess_database flag is active (1)
                            0,#intent
                            0,#sentiment
                            0,#epitome
                            0,#vad lexicon
                            1,#length
                            0,#emotion 32
                            0,#emotion 20
                            0,#emotion 8
                            0,#emotion mimicry
                            1,#reduced empathy labels
                            1, #exchange number
                            1, #output processed database
                            0 #7 emotion labels
                            ]


control_vector_dictionary = {0:'database_to_classify', 1: 'intent', 2:'sentiment', 3:'EPITOME Mechanisms', 4:'VAD vectors', 5: 'Length of utterances', 6: '32 emotion labels', 7:'20 emotion labels', 8: '8 emotion labels', 9: 'emotion mimicry', 10: 'Reduced empathy labels', 11: 'exchange number', 12: 'output', 13: '7_emotion_labels'}

feature2number = {'database_to_classify':0,'intent' : 1, 'sentiment' : 2, 'epitome':3, 'VAD_vectors':4, 'utterance_length':5, '32_emotion_labels':6,'20_emotion_labels':7, '8_emotion_labels':8, 'emotion_mimicry':9, 'Reduce_empathy_labels':10, 'exchange_number' : 11, 'output' : 12,'7_emotion_labels': 13}

#print(database_control_vector[feature2number['exchange_number']])

if auto_experiments == 1:
    #select number of features to modify
    number_of_features=4
    #create list of variations
    variation_lst = list(map(list, itertools.product([0, 1], repeat=number_of_features)))
    #control vectors that will be carried out automatically
    control_vector_list = []
    for i in list(variation_lst): 
        control_vector = [database_control_vector[0]]
        control_vector = [database_control_vector[1]]
        control_vector = [database_control_vector[2]]
        control_vector = [database_control_vector[3]] 
        control_vector = [database_control_vector[4]]                               
        control_vector = control_vector + i #feature changing
        control_vector.append(database_control_vector[9]) #mimicry
        control_vector.append(database_control_vector[10]) #reduced empathy labels
        control_vector.append(database_control_vector[11]) #exchange_number
        control_vector.append(database_control_vector[12]) #output processed dataframe       
    print('List of control vectors created, auto_experimentation is on.')
    print(f'Number of features that will vary: {number_of_features}')
    print(f'Following experiments will carried out: {experiment_number} to {experiment_number+len(control_vector_list)}')
    answer = input("Continue (y/n)?")
    if answer.lower() in ["y","yes"]:
        print('Automatic experimentation will be carried out, directories might be overwritten')
    elif answer.lower() in ["n","no"]:
        print('Aborting operation')
        sys.exit(0)
    else:
        print('Wrong input received, aborting operation')
        sys.exit(1)
else: 
    #control_vectors that will be 
    print('Single experiment mode selected')
    control_vector_list = [database_control_vector]



for control_vector in control_vector_list:
    print(f'Experiment #{experiment_number}')
    #create a string version of features
    features_used = 'Features used: \n\n'
    for i in range(len(control_vector)):
        features_used  = features_used + control_vector_dictionary[i] + ': ' + str(control_vector[i]) + ' \n'


    print(features_used)

    #If it is necessary to reprocess database, send instructions and carry out the procedure. 
    if reprocess_database == 1:
        if test_database_dir == train_database_dir:
            if 'EmpatheticExchanges' in train_database_dir:
                print('Processing EmpatheticExchanges')
                control_vector[0] = 1
                print(control_vector)
                data_processer.process_database(control_vector)
            else:
                print('Processing EmpatheticConversations in exchange format')
                control_vector[0] = 0
                print(control_vector)
                data_processer.process_database(control_vector)           
        else: 
            print('Processing both databases')
            control_vector[0] = 0
            data_processer.process_database(control_vector)
            control_vector[0] = 1
            data_processer.process_database(control_vector)
    else:
        print('No database reprocessing selected, carrying out with database in the "processed databases" folder ')

    print('')

    #Create path for experiment output
    results_path = current_dir + '/Experiments/outputs/Experiment ' + str(experiment_number) + '/'
    if os.path.exists(results_path):
        print('WARNING: Experimental path found')
    else:
        print('Creating Experimental path')
        os.mkdir(results_path)

    print('')

    #If you have to retrain the model, obtain training dataset
    if do_training == 1:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        trainFile = current_dir + train_database_dir + 'train.csv'
        data_train = pd.read_csv(trainFile)
        data_train["empathy"] = data_train["empathy"].astype('int')
        data_train["empathy"] = data_train["empathy"].astype('string')
        if control_vector[feature2number['emotion_mimicry']] == 1:
            data_train['mimicry'] = data_train['mimicry'].astype('category')
            data_train['mimicry'] = data_train['mimicry'].astype('string')
        if (control_vector[feature2number['32_emotion_labels']] == 1) or (control_vector[feature2number['20_emotion_labels']] == 1) or (control_vector[feature2number['8_emotion_labels']] == 1) :
            data_train['speaker_emotion'] = data_train['speaker_emotion'].astype('category')
            data_train['listener_emotion'] = data_train['listener_emotion'].astype('category')


        #modify features last minute
        #data_train = data_train.drop(columns=['dominance_speaker'])
        #data_train = data_train.drop(columns=['dominance_listener'])
        #data_train = data_train.drop(columns=['predictions_EX'])
        #data_train = data_train.drop(columns=['predictions_IP'])



        #data_train['empathy_red'] = data_train.apply(lambda x: 1 if (x['empathy'] == 2 or x['empathy'] == 1)  else 2, axis = 1)
        #data_train = data_train.drop(columns=['empathy'])
        #data_train = data_train.rename(columns={"empathy_red": "empathy"})



        print(f'Features from the training database')
        print(data_train.columns)
        print(f'Number of datapoints in training database: {len(data_train)}')
    #Else, use an already trained model at a given path.
    else: 
        print(f'NO TRAINING, using model located at: {already_trained_model_path}')

    #Where the test file is located
    testFile = current_dir + test_database_dir + 'test.csv'
    #read test dataframe
    data_test = pd.read_csv(testFile)
    

    #modify features last minute    
    #data_test = data_test.drop(columns=['dominance_speaker'])
    #data_test = data_test.drop(columns=['dominance_listener'])
    #data_test = data_test.drop(columns=['predictions_EX'])
    #data_test = data_test.drop(columns=['predictions_IP']) 

    #data_test['empathy_red'] = data_test.apply(lambda x: 1 if (x['empathy'] == 2 or x['empathy'] == 1)  else 2, axis = 1)
    #data_test = data_test.drop(columns=['empathy'])
    #data_test = data_test.rename(columns={"empathy_red": "empathy"})



    #Print features of the dataframe used for testing
    print(f'Features from the testing database')
    print(data_test.columns)



    #Print ho wmany datapoints 
    print(f'Number of datapoints in testing database: {len(data_test)}')

    print()

    #If we have to train, carry out the procedure using a trainer and test the results using a test set 
    if do_training == 1:
        trainer.train(experiment_number,data_train)
        model_path = current_dir + '/Experiments/outputs/Experiment '+ str(experiment_number) + '/' + 'trained_pbc4cip.sav'
        tester.test(experiment_number,data_test,model_path,features_used)
    else: 
    #If training is unnecessary, test the classifier on a test set 
        tester.test(experiment_number,data_test,already_trained_model_path,features_used)
    #increase experiment number
    print(f'Successfully carried out experiment # {experiment_number}')

    experiment_number +=1
    print()
    print()







