from math import isnan
import os, os.path
import pandas as pd
import torch
from torch import cuda
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer,pipeline
import re
import numpy as np
import math
from numpy.linalg import norm

#sentiment
from classifiers.sentiment import sentiment_prediction as sp
#intent
from classifiers.empathetic_intent import intent_prediction as ip
#epitome 
#from classifiers.epitome_mechanisms import epitome_predictor as epitome
from classifiers.epitome_mechanisms import epitome_predictor as epitome
from classifiers.nrc_vad_lexicon import lexicon_analysis as lexicon
from sklearn.model_selection import train_test_split
#emotion
from classifiers.course_grained_emotion import pretrained_32emotions as em32
from classifiers.course_grained_emotion import pretrained_7emotions as em7
from classifiers.course_grained_emotion import emotion_reductor as em_red
import contractions
from spellchecker import SpellChecker
import regex



emotion_dictionary = {0: 'afraid', 1: 'angry', 2: 'annoyed', 3: 'anticipating', 4: 'anxious', 5: 'apprehensive', 6: 'ashamed', 7: 'caring', 8: 'confident', 9: 'content', 
 10: 'devastated', 11: 'disappointed', 12: 'disgusted', 13: 'embarrassed', 14: 'excited', 15: 'faithful', 16: 'furious', 17: 'grateful', 18: 'guilty', 
 19: 'hopeful', 20: 'impressed', 21: 'jealous', 22: 'joyful', 23: 'lonely', 24: 'nostalgic', 25: 'prepared', 26: 'proud', 27: 'sad', 28: 'sentimental', 
 29: 'surprised', 30: 'terrified', 31: 'trusting'}
intent_labels = ['agreeing','acknowledging','encouraging','consoling','sympathizing','suggesting','questioning','wishing','neutral']

ekman_emo_dic = {0:'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise' }


def expand_contractions(text):
    if "'" in str(text):
        expanded_words = []    
        for word in text.split():
            expanded_words.append(contractions.fix(word))   
        expanded_text = ' '.join(expanded_words)
        return str(expanded_text)
    else: 
        return str(text)

def clean_string(text, spell):
    punc_re = regex.compile(r'(\p{Punctuation})')    
    text = str(text)
    text = re.sub("_comma_", ',', text)
    text = re.sub("[^0-9a-zA-Z?!,.]+", ' ', text)
    text = text.lower()
    arr = punc_re.split(text)
    #print(arr)
    for i in range(len(arr)):
      sub = arr[i].split()
      misspelled = spell.unknown(sub)
      for j in range(len(sub)):
        if sub[j] in misspelled:
          #print(f'mispelled {arr2[j]}')
          if spell.correction(sub[j]) != None:
              sub[j] = spell.correction(sub[j]) 
        arr[i] = ' '.join(sub)
      clean_text = ' '.join(arr)
      clean_text = re.sub(' +', ' ',clean_text)
    return clean_text



def get_emp_intent(dataframe_row,mdl,tokenzr,dev):
    #dev = device
    if dataframe_row['is_response'] > 0:
        intent = dataframe_row['utterance']
        #print(dataframe_row['utterance'])
        intent = ip.get_empathetic_intent(str(dataframe_row['utterance']),mdl,tokenzr,dev)
        #print(dataframe_row['utterance'])
    else:
        intent = -1
    return intent

def get_emp_intent_probabilities(dataframe_row,mdl,tokenzr,dev,utt_column):
    #dev = device
    if dataframe_row['is_response'] > 0:
        intent = dataframe_row['utterance']
        #print(dataframe_row['utterance'])
        intent = ip.get_empathetic_intent_logits(str(dataframe_row[utt_column]),mdl,tokenzr,dev)
        #print(dataframe_row['utterance'])
    else:
        intent = -1
    return intent

def get_sentiment_probabilities(dataframe_row,mdl,tokenzr,utt_column):
    #gets the probabilities of each sentiment label
    sentiment_lst = sp.get_sentiment(str(dataframe_row[utt_column]),mdl,tokenzr)
    return sentiment_lst

def get_emotion_probabilities(dataframe_row,mdl,tokenzr,utt_column):
    #get the probabilities of each of the 32 emotions
    emotion_lst = em32.get_emotion_32(str(dataframe_row[utt_column]),mdl,tokenzr)
    return emotion_lst

def get_emotion_label(dataframe_row,mdl,tokenzr,utt_column):
    #get the one emotion label out of 32 for a text utterance
    emotion_lst = em32.get_emotion_32(str(dataframe_row[utt_column]),mdl,tokenzr)
    label = emotion_dictionary[np.argmax(emotion_lst)]
    return label

def get_emotion_label_7(dataframe_row,mdl,tokenzr,utt_column):
    #get the one emotion label out of 32 for a text utterance
    emotion_lst = em7.get_emotion_7(str(dataframe_row[utt_column]),mdl,tokenzr)
    label = ekman_emo_dic[np.argmax(emotion_lst)]
    return label

def get_word_len(utterance):
    utterance = str(utterance)
    utterance = re.sub("_comma_", ',', utterance)
    arr = utterance.split()
    #string2clean = re.sub("[^0-9a-zA-Z']+", ' ', string2clean)
    return len(arr)

def get_sentiment_label(dataframe_row,mdl,tokenzr,utt_column):
    #gets the sentiment in accordance to the label we want
    #0 - negative, 1 - neutral, 2 - positive
    label_val = ['negative','neutral', 'positive']
    sentiment_lst = sp.get_sentiment(str(dataframe_row[utt_column]),mdl,tokenzr)
    #print(sentiment_lst)
    index_max = max(range(len(sentiment_lst)), key=sentiment_lst.__getitem__)
    return label_val[index_max]

def is_responde(utterance_id):
    return int((utterance_id %2 == 0))

def fill_dataframe(dataframe):
    #This method fills empty spaces in the dataframe based on the conversation id, context, and prompt. 
    #Additionally, adds back 'utterance_idx' that says which utterance within the conversation it is, and is_response, which says if it is a response
    dataframe['utterance_idx'] = 0
    utt_idx = 0
    current_conv_id = ''
    current_context = ''
    current_evaluation = 0
    current_prompt = ''
    for i in range(len(dataframe)):
        if (str(dataframe.loc[i, 'conv_id']) != 'nan') and (str(dataframe.loc[i, 'context']) != 'nan') and (str(dataframe.loc[i, 'prompt']) != 'nan') and (str(dataframe.loc[i, 'conv_id']) != ' ') and (str(dataframe.loc[i, 'context']) != ' ') and (str(dataframe.loc[i, 'prompt']) != ' '):
            dataframe.loc[i, 'utterance_idx'] = 1
            utt_idx = 1
            current_conv_id = dataframe.loc[i, 'conv_id']
            current_context = dataframe.loc[i, 'context']
            current_prompt = dataframe.loc[i, 'prompt']
            current_evaluation = dataframe.loc[i, 'evaluation']
        else:
            utt_idx+=1
            dataframe.loc[i, 'conv_id'] = current_conv_id
            dataframe.loc[i, 'context'] = current_context
            dataframe.loc[i, 'prompt'] = current_prompt
            dataframe.loc[i, 'evaluation'] = current_evaluation
            dataframe.loc[i, 'utterance_idx'] = utt_idx
            #print(dataframe['conv_id'][i])
    
    dataframe['is_response'] = dataframe['utterance_idx'].apply(is_responde)
    return dataframe

def modify_to_exchange_format(dataframe):
    dataframe['speaker_utterance'] = ''
    dataframe['listener_utterance'] = ''
    #exchange number in the conversation
    dataframe['exchange_number'] = 0
    #dataframe['speaker_sentiment'] = ''
    #dataframe['listener_sentiment'] = ''
    conversation_ids = dataframe.conv_id.unique()
    epitome_df = pd.DataFrame()
    for i in conversation_ids:
        convo = dataframe[dataframe['conv_id'] == str(i)]

        #Ignore the last extra utterance from the speaker, it is unnecessary
        if len(convo)%2 != 0:
            convo = convo[:-1]
        #For every utterance in the index, if it is a "listener post", we get the exchange and annotate it. 
        for i in convo.index:
            if(convo.loc[i,'utterance_idx'] %2 == 0):
                convo.loc[i, 'speaker_utterance'] = convo.loc[i-1, 'utterance']
                convo.loc[i, 'listener_utterance'] = convo.loc[i, 'utterance']
                convo.loc[i, 'exchange_number'] = int(convo.loc[i,'utterance_idx'] / 2)
                #convo.loc[i, 'speaker_sentiment'] = convo.loc[i-1, 'sentiment_label']
                #convo.loc[i, 'listener_sentiment'] = convo.loc[i, 'sentiment_label']
        epitome_df = pd.concat([epitome_df,convo])

    epitome_df = epitome_df[epitome_df['is_response'] != 0]
    dfcolumns = dataframe.columns.to_list()
    dfcolumns.remove('speaker_utterance')
    dfcolumns.remove('listener_utterance')
    #print(dfcolumns)
    #epitome_df = epitome_df.drop(columns=['utterance','speaker_idx','utterance_idx','is_response','sentiment_label'])
    epitome_df = epitome_df.drop(columns=['utterance','speaker_idx','utterance_idx','is_response'])
    epitome_df = epitome_df.reset_index(drop=True)
    #print(epitome_df.head())
    return epitome_df

def get_VAD_values_for_both(dataframe):
    #setup lexicon utilities
    lexicon_df,wnl,stp_wrds = lexicon.setup_lexicon('classifiers/nrc_vad_lexicon/BipolarScale/NRC-VAD-Lexicon.txt')
    dataframe['vad_speaker'] = dataframe['speaker_utterance'].apply(lexicon.get_avg_vad, args = (lexicon_df,wnl,stp_wrds)) 
    dataframe['vad_listener'] = dataframe['listener_utterance'].apply(lexicon.get_avg_vad, args = (lexicon_df,wnl,stp_wrds)) 
    dataframe[['valence_speaker','arousal_speaker','dominance_speaker']] = pd.DataFrame(dataframe.vad_speaker.tolist(),index = dataframe.index)
    dataframe[['valence_listener','arousal_listener','dominance_listener']] = pd.DataFrame(dataframe.vad_listener.tolist(),index = dataframe.index)
    dataframe = dataframe.drop(columns = ['vad_speaker','vad_listener'])
    #print(vad)
    return dataframe

def get_cosine_similarity(dataframe_row):

    speaker_va_vector = np.array([dataframe_row['valence_speaker'], dataframe_row['arousal_speaker']])
    listener_va_vector = np.array([dataframe_row['valence_listener'], dataframe_row['arousal_listener']])
    if (np.any(speaker_va_vector) == False) or (np.any(listener_va_vector) == False):
        cosine = -1
    else:
        cosine = np.dot(speaker_va_vector,listener_va_vector)/(norm(speaker_va_vector)*norm(listener_va_vector))
    return cosine


def process_database(control_vector):
    print('Starting database processing!')
    #dictionary to obtain position in binary control vector using the feature of interest
    feature2number = {'database_to_classify':0,'intent' : 1, 'sentiment' : 2, 'epitome':3, 'VAD_vectors':4, 'utterance_length':5, '32_emotion_labels':6,
                      '20_emotion_labels':7, '8_emotion_labels':8, 'emotion_mimicry':9, 'Reduce_empathy_labels':10, 'exchange_number' : 11, 
                      'output':12,'7_emotion_labels': 13}


    '''
    control_vector = [
                  1,    #database to classify 0 = empatheticconversations (old), 1 empatheticexchanges (new) 
                  1,    #intent
                  1,    #sentiment
                  1,    #epitome
                  1,    #vad lexicon
                  1,    #length
                  1,    #emotion 32
                  0,    #emotion 20
                  0,    #emotion 8
                  0,    #emotion mimicry
                  0,    #reduced_empathy_labels
                  0,     #exchange number in the conversation
                  1      #output processed database
                  ]
    '''

    if control_vector[feature2number['database_to_classify']] == 0:
        database_to_process = 'EmpatheticConversations-EC'
    else:
        database_to_process = 'data_samples'

    #setup subdirectory of processed data
    dataSubDir = './unprocessed_databases/'+database_to_process+'/'
    empIntSubDir = './classifiers/empathetic_intent/'


    #get all files
    file_list = [name for name in os.listdir(dataSubDir) if os.path.isfile(dataSubDir+name)]
    #create empty dataframe
    df = pd.DataFrame()

    #data retrieval
    if database_to_process == 'data_samples':    
    #get all datasets, process them, and join them
        print('reading datasets....')
        for file in file_list:
            #print(file)
            temp_df = pd.read_excel(dataSubDir+file, engine="odf")
            #set up from format given to evaluators to full dataframe
            temp_df = fill_dataframe(temp_df)
            #concatenate the datasets
            df = pd.concat([df,temp_df])
            df.reset_index(drop=True, inplace=True)
        print('done')
        #Check if there are any bad evaluations.
        if len(df[df['evaluation'].isin([1,2,3,4,5]) == False]) > 0:
            print('Error: Database contains bad evaluations, manually check the following conversations')
            print(df[df['evaluation'].isin([1,2,3,4,5]) == False])
            exit(1)
        df = df.rename(columns={"evaluation": "empathy"})
    else:
        #get the dataset EmpatheticConversations (400 conversations from empatheticdialogues evaluated using the Delphi method)
        print('retrieving dataset....')
        temp_df = pd.read_csv(dataSubDir+'EmpatheticConversations.csv')
        #print(temp_df.head())
        df = pd.concat([df,temp_df])
        df['is_response'] = df['utterance_idx'].apply(is_responde)
        df = df.drop(columns=['ut_len','Talker','Sentiment','Emotion','Taxonomy','Intent'])
        df = df.rename(columns={"Empathy": "empathy"})
        print('done')
      
    #This is to test with a small dataframe
    #df = df.loc[0:200]

    print('Expanding contractions....')
    df['utterance'] = df['utterance'].apply(expand_contractions)
    print('done')

    spellcheck = SpellChecker()

    #print('Cleaning string....')
    #df['utterance'] = df['utterance'].apply(clean_string, args=(spellcheck,))
    #print('done')

    #get empathetic intent
    if control_vector[feature2number['intent']] == 1:
        print('getting intent....')
        model,tokenizer,device = ip.loadModelTokenizerAndDevice(empIntSubDir) #get model and parameters
        df['empathetic_intent'] = df.apply(get_emp_intent_probabilities, axis=1, args = (model,tokenizer,device,'utterance'))
        #exchange_df[intent_labels] = pd.DataFrame(exchange_df.empathetic_intent.tolist(),index = df.index)
        #exchange_df[intent_labels] = pd.DataFrame(exchange_df.empathetic_intent.tolist(),index = exchange_df.index)
        #df['empathetic_intent'] = df.apply(get_emp_intent, axis=1, args = (model,tokenizer,device))  #apply empathetic intent extraction
        print('done')

   
    #prepare the database in exchange format
    print('preparing database in exchange format....')
    exchange_df = modify_to_exchange_format(df)
    print('done')


    #sentiment labels
    if control_vector[feature2number['sentiment']] == 1:
        print('getting sentiment....')
        sent_model, sent_tokenzr = sp.loadSentimentModel() #get model and tokenizer
        exchange_df['speaker_sentiment'] = exchange_df.apply(get_sentiment_probabilities,axis = 1, args = (sent_model,sent_tokenzr,'speaker_utterance')) #apply sentiment label extraction to speaker
        exchange_df[['s_negative','s_neutral', 's_positive']] = pd.DataFrame(exchange_df.speaker_sentiment.tolist(),index = exchange_df.index)
        exchange_df['listener_sentiment'] = exchange_df.apply(get_sentiment_probabilities,axis = 1, args = (sent_model,sent_tokenzr,'listener_utterance')) #apply sentiment label extraction to speaker
        exchange_df[['l_negative','l_neutral', 'l_positive']] = pd.DataFrame(exchange_df.listener_sentiment.tolist(),index = exchange_df.index)
        exchange_df = exchange_df.drop(columns=['speaker_sentiment','listener_sentiment'])
        print('done')


    #call the epitome classifier and get the epitome mechanisms: Emotional reaction (ER), Intepretation (IP), and Explorations (EX)
    if control_vector[feature2number['epitome']] == 1:
        print('getting EPITOME mechanisms....')
        exchange_df = epitome.predict_epitome_values('classifiers/epitome_mechanisms/trained_models',exchange_df)
        #exchange_df = exchange_df.drop(columns=['predictions_EX'])
        #exchange_df = exchange_df.drop(columns=['predictions_IP'])
        #exchange_df = exchange_df.drop(columns=['predictions_EM'])
        print('done')

    #Annotate Valence, Arousal, and Dominance for the speaker and listener utterances.
    if control_vector[feature2number['VAD_vectors']] == 1:
        print('Annotating VAD values.....')
        exchange_df = get_VAD_values_for_both(exchange_df)
        print('done')

    #Get length of utterances
    if control_vector[feature2number['utterance_length']] == 1:
        print('getting length of utterances....')
        exchange_df['s_word_len'] = exchange_df['speaker_utterance'].apply(get_word_len) 
        exchange_df['l_word_len'] = exchange_df['listener_utterance'].apply(get_word_len) 
        print('done')

    #separate intent 
    if control_vector[feature2number['intent']] == 1:
        print('separating intent....')
        exchange_df[intent_labels] = pd.DataFrame(exchange_df.empathetic_intent.tolist(),index = exchange_df.index)
        exchange_df = exchange_df.drop(columns=['empathetic_intent'])
        print('done')

    #get 32 course-grained-emotion labels
    if control_vector[feature2number['32_emotion_labels']] == 1:      
        print('getting 32 emotion labels.....')
        emo32_model, emo32_tokenzr = em32.load32EmotionsModel() #get model and tokenizer
        exchange_df['speaker_emotion'] = exchange_df.apply(get_emotion_label,axis = 1, args = (emo32_model,emo32_tokenzr,'speaker_utterance')) #apply emotion label extraction to speaker
        exchange_df['listener_emotion'] = exchange_df.apply(get_emotion_label,axis = 1, args = (emo32_model,emo32_tokenzr,'listener_utterance')) #apply emotion label extraction to listener
        #reduce number of emotion labels
        print('done')

    #get 20 course-grained-emotion labels
    if control_vector[feature2number['20_emotion_labels']] == 1:
        print('getting 20 emotion labels....')
        emo32_model, emo32_tokenzr = em32.load32EmotionsModel() #get model and tokenizer
        exchange_df['speaker_emotion'] = exchange_df.apply(get_emotion_label,axis = 1, args = (emo32_model,emo32_tokenzr,'speaker_utterance')) #apply emotion label extraction to speaker
        exchange_df['listener_emotion'] = exchange_df.apply(get_emotion_label,axis = 1, args = (emo32_model,emo32_tokenzr,'listener_utterance')) #apply emotion label extraction to listener
        exchange_df = em_red.reduce_emotion_labels('speaker_emotion',exchange_df)
        exchange_df = em_red.reduce_emotion_labels('listener_emotion',exchange_df)
        print('done')

    #get 8 course-grained-emotion labels
    if control_vector[feature2number['8_emotion_labels']] == 1:
        print('getting 8 emotion labels....')
        emo32_model, emo32_tokenzr = em32.load32EmotionsModel() #get model and tokenizer
        exchange_df['speaker_emotion'] = exchange_df.apply(get_emotion_label,axis = 1, args = (emo32_model,emo32_tokenzr,'speaker_utterance')) #apply emotion label extraction to speaker
        exchange_df['listener_emotion'] = exchange_df.apply(get_emotion_label,axis = 1, args = (emo32_model,emo32_tokenzr,'listener_utterance')) #apply emotion label extraction to listener
        exchange_df = em_red.reduce_emotion_labels_to_8('speaker_emotion',exchange_df)
        exchange_df = em_red.reduce_emotion_labels_to_8('listener_emotion',exchange_df)
        print('done')

    #get if mimicry is being done
    if control_vector[feature2number['emotion_mimicry']] == 1:
        print('getting mimicry.........')
        if(control_vector[4] == 1):
            #get the emotional distance and if it is less than 0.1 set mimicry to 1
            print('Obtaining mimicry through emotional distance using VAD....')
            exchange_df['emotional_similarity'] = exchange_df.apply(get_cosine_similarity,axis = 1) #obtain cosine similarity between valence and arousal vector
            exchange_df['mimicry'] = exchange_df.apply(lambda x: 1 if x['emotional_similarity'] > 0.7 else 0, axis = 1)
            exchange_df = exchange_df.drop(columns = ['emotional_similarity'])
        else: 
            #Else, obtain the least amount of emotion labels possible and use that to get the mimicry
            print('No VAD values , obtaining mimicry through emotional distance using newly created VAD....')
            emo32_model, emo32_tokenzr = em32.load32EmotionsModel() #get model and tokenizer
            #print('Annotating VAD values.....')
            exchange_df = get_VAD_values_for_both(exchange_df)
            exchange_df['emotional_similarity'] = exchange_df.apply(get_cosine_similarity,axis = 1) #obtain cosine similarity between valence and arousal vector
            exchange_df['mimicry'] = exchange_df.apply(lambda x: 1 if x['emotional_similarity'] > 0.7 else 0, axis = 1)
            exchange_df = exchange_df.drop(columns = ['valence_speaker','arousal_speaker','dominance_speaker','valence_listener','arousal_listener','dominance_listener','emotional_similarity'])
        print('done')
    
    #reduce empathy labels
    if control_vector[feature2number['Reduce_empathy_labels']] == 1:
        print('Reducing labels to three....')
        exchange_df['empathy_red'] = exchange_df.apply(lambda x: 1 if (x['empathy'] == 2 or x['empathy'] == 1)  else (2 if x['empathy'] == 3 else 3), axis = 1)
        exchange_df = exchange_df.drop(columns=['empathy'])
        exchange_df = exchange_df.rename(columns={"empathy_red": "empathy"})
        print('done!')

    #if explicitely told to ignore exchange number
    if control_vector[feature2number['exchange_number']] == 0:
        print('eliminating exchange_number')
        exchange_df = exchange_df.drop(columns=['exchange_number'])
    else: 
        print('Keeping exchange number')

    #get 32 course-grained-emotion labels
    if control_vector[feature2number['7_emotion_labels']] == 1:      
        print('getting 7 emotion labels.....')
        emo7_model, emo7_tokenzr = em7.load7EmotionsModel() #get model and tokenizer
        exchange_df['speaker_emotion'] = exchange_df.apply(get_emotion_label_7,axis = 1, args = (emo7_model,emo7_tokenzr,'speaker_utterance')) #apply emotion label extraction to speaker
        exchange_df['listener_emotion'] = exchange_df.apply(get_emotion_label_7,axis = 1, args = (emo7_model,emo7_tokenzr,'listener_utterance')) #apply emotion label extraction to listener
        #reduce number of emotion labels
        print('done')


    if control_vector[feature2number['output']] == 1:
        print('separating dataframe for classification...')
        X = exchange_df.drop(columns=['empathy'])
        y = exchange_df['empathy']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42,stratify=y)
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        print('done')
        #Output database in csv format. 
        if database_to_process == 'data_samples':  
            #output the processed database
            exchange_df.to_csv('./processed_databases/EmpatheticExchanges/EmpatheticExchanges.csv',index=False)
            train_df.to_csv('./processed_databases/EmpatheticExchanges/EmpatheticExchanges_train.csv',index=False)
            test_df.to_csv('./processed_databases/EmpatheticExchanges/EmpatheticExchanges_test.csv',index=False)
            columns_to_drop = ['conv_id','prompt','speaker_utterance', 'listener_utterance','context']
            train_df = train_df.drop(columns=columns_to_drop)
            test_df = test_df.drop(columns=columns_to_drop)
            train_df.to_csv('./processed_databases/EmpatheticExchanges/train.csv',index=False)
            test_df.to_csv('./processed_databases/EmpatheticExchanges/test.csv',index=False)
        else:
            exchange_df.to_csv('./processed_databases/EmpatheticConversationsExchangeFormat/EmpatheticConversations_ex.csv',index=False)
            train_df.to_csv('./processed_databases/EmpatheticExchanges/EmpatheticExchanges_train.csv',index=False)
            test_df.to_csv('./processed_databases/EmpatheticExchanges/EmpatheticExchanges_test.csv',index=False)
            columns_to_drop = ['conv_id','prompt','speaker_utterance', 'listener_utterance','context']
            train_df = train_df.drop(columns=columns_to_drop)
            test_df = test_df.drop(columns=columns_to_drop)
            train_df.to_csv('./processed_databases/EmpatheticConversationsExchangeFormat/train.csv',index=False)
            test_df.to_csv('./processed_databases/EmpatheticConversationsExchangeFormat/test.csv',index=False)
    
    print('Database processed successfully!')



if __name__ == '__main__':

    main()