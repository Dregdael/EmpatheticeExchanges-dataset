

import pickle
import pandas as pd
import torch
import os
import sys
import random 
import re
#import classifier
from PBC4cip import PBC4cip
from PBC4cip.core.Evaluation import obtainAUCMulticlass
from PBC4cip.core import Dataset as pbcDataset


#utilities for database management
import numpy as np
import pandas as pd
import os
import argparse

import train_classifier as trainer
import test_classifier as tester
import database_processing_package as data_processer

#relevant classifiers for annotating exchange feature
from classifiers.empathetic_intent import intent_prediction as ip
from classifiers.sentiment import sentiment_prediction as sp
from classifiers.epitome_mechanisms import epitome_predictor as epitome
from classifiers.nrc_vad_lexicon import lexicon_analysis as lexicon
from classifiers.course_grained_emotion import pretrained_32emotions as em32
from classifiers.course_grained_emotion import emotion_reductor as em_red

from spellchecker import SpellChecker




def load_supplementary_classifiers(att_lst): 
    sentiment_flag = 0
    ex_num_flag = 0
    epitome_flag = 0
    vad_flag = 0
    intent_flag = 0
    mimicry_flag = 0
    length_flag = 1 #always on
    emo_labels_flag = 0

    model_components = []

    #get the classifiers necessary for processing the response
    if {'s_negative','s_neutral','s_positive','l_negative','l_neutral','l_positive'} & set(att_lst): 
        sentiment_flag = 1
        empIntSubDir = './classifiers/empathetic_intent/'
        sent_model, sent_tokenzr = sp.loadSentimentModel() #get model and tokenizer
        model_components.extend([sent_model,sent_tokenzr])
    else:
        model_components.extend([0,0])
    if {'exchange_number'} & set(att_lst): 
        ex_num_flag = 1
    if {'predictions_ER','predictions_IP','predictions_EX'} & set(att_lst): 
        epitome_flag = 1
        epitome_empathy_classifier = epitome.load_epitome_classifier('classifiers/epitome_mechanisms/trained_models')
        model_components.append(epitome_empathy_classifier)
    else:
        model_components.append(0)
    if {'valence_speaker','arousal_speaker','dominance_speaker','valence_listener','arousal_listener','dominance_listener'} & set(att_lst): 
        vad_flag = 1
        lexicon_df, wnl, stp_wrds = lexicon.setup_lexicon('classifiers/nrc_vad_lexicon/BipolarScale/NRC-VAD-Lexicon.txt')
        model_components.extend([lexicon_df, wnl, stp_wrds])
    else:
        model_components.extend([0,0,0])
    if {'agreeing','acknowledging','encouraging','consoling','sympathizing',
        'suggesting','questioning','wishing','neutral'} & set(att_lst): 
        intent_flag = 1
        empIntSubDir = './classifiers/empathetic_intent/'
        model_intent,tokenizer_intent,device = ip.loadModelTokenizerAndDevice(empIntSubDir) #get model and parameters
        model_components.extend([model_intent,tokenizer_intent,device])
    else:
        model_components.extend([0,0,0])
    if {'mimicry'} & set(att_lst): 
        mimicry_flag = 1
        lexicon_df, wnl, stp_wrds = lexicon.setup_lexicon('classifiers/nrc_vad_lexicon/BipolarScale/NRC-VAD-Lexicon.txt')
        model_components.extend([lexicon_df, wnl, stp_wrds])   
    else:
        model_components.extend([0, 0, 0])   
    if {'speaker_emotion','listener_emotion'} & set(att_lst):
        emo_labels_flag = 1
        emo32_model, emo32_tokenzr = em32.load32EmotionsModel()
        model_components.extend([emo32_model, emo32_tokenzr])
    else:
        model_components.extend([0,0])
    flag_array = [sentiment_flag,ex_num_flag,epitome_flag,vad_flag,intent_flag,mimicry_flag,length_flag,emo_labels_flag]

    return flag_array, model_components




def judge_exchange(classifier,flag_array, att_lst,speaker_utterance,listener_utterance,model_components,role):
    processed_exchange, pred  = predict_exchange_empathy(classifier, flag_array, 1, att_lst,speaker_utterance, listener_utterance,model_components)
    print(f"Empathy prediction for {role}: {pred}/3")
    if pred < 2: 
        print('Low empathy detected!')
        recommendation = get_recommentation(classifier, processed_exchange,role)
        print(f"interjection by Haru: {recommendation}")
        #print(recommendation)
        return False
    return True

#Main processing function

def predict_exchange_empathy_source(classifier, flag_array,ex_num, att_lst, speaker_utterance, listener_utterance, sent_model, sent_tokenzr, epitome_empathy_classifier, lexicon_df, wnl, stp_wrds, model_intent,tokenizer_intent,device,lexicon_df_m, wnl_m, stp_wrds_m,emo32_model,emo32_tokenzr):
    #Turn an string exchange into a dataframe
   #print(att_lst)
    dummy_dic = {feature: [0] for feature in att_lst}
    #for i in range(len(att_lst)):
    #    #print(att_lst[i])
    #    dummy_dic.update({att_lst[i]: [0]})
    
    exchange_df = pd.DataFrame.from_dict(dummy_dic)
    exchange_df['speaker_utterance'] = speaker_utterance
    exchange_df['listener_utterance'] = listener_utterance

    #Add the features to the exchange dataframe according to the ones used by model. We use the same functions as the experiments, which is why we do everything through dataframes
    if flag_array[0] == 1: 
        #get sentiment
        exchange_df['speaker_sentiment'] = exchange_df.apply(data_processer.get_sentiment_probabilities,axis = 1, args = (sent_model,sent_tokenzr,'speaker_utterance')) 
        exchange_df[['s_negative','s_neutral', 's_positive']] = pd.DataFrame(exchange_df.speaker_sentiment.tolist(),index = exchange_df.index)
        exchange_df['listener_sentiment'] = exchange_df.apply(data_processer.get_sentiment_probabilities,axis = 1, args = (sent_model,sent_tokenzr,'listener_utterance')) 
        exchange_df[['l_negative','l_neutral', 'l_positive']] = pd.DataFrame(exchange_df.listener_sentiment.tolist(),index = exchange_df.index)
        exchange_df = exchange_df.drop(columns=['speaker_sentiment','listener_sentiment'])
    if flag_array[1] == 1:
        #set exchange number to the one we provide
        exchange_df['exchange_number'] = ex_num
    if flag_array[2] == 1:
        #get epitome communication mechanisms
        exchange_df = epitome.classify_epitome_values(epitome_empathy_classifier, exchange_df)
    if flag_array[3] == 1:
        #get VAD vectors
        exchange_df['vad_speaker'] = exchange_df['speaker_utterance'].apply(lexicon.get_avg_vad, args = (lexicon_df,wnl,stp_wrds)) 
        exchange_df['vad_listener'] = exchange_df['listener_utterance'].apply(lexicon.get_avg_vad, args = (lexicon_df,wnl,stp_wrds)) 
        exchange_df[['valence_speaker','arousal_speaker','dominance_speaker']] = pd.DataFrame(exchange_df.vad_speaker.tolist(),index = exchange_df.index)
        exchange_df[['valence_listener','arousal_listener','dominance_listener']] = pd.DataFrame(exchange_df.vad_listener.tolist(),index = exchange_df.index)
        exchange_df = exchange_df.drop(columns = ['vad_speaker','vad_listener'])
    if flag_array[4] == 1:
        #get word lengths
        exchange_df['s_word_len'] = exchange_df['speaker_utterance'].apply(data_processer.get_word_len) 
        exchange_df['l_word_len'] = exchange_df['listener_utterance'].apply(data_processer.get_word_len) 
    if flag_array[5] == 1: 
        #get intent
        exchange_df['utterance'] = exchange_df['listener_utterance']
        exchange_df['is_response'] = 1
        exchange_df['empathetic_intent'] = exchange_df.apply(data_processer.get_emp_intent_probabilities, axis=1, args = (model_intent,tokenizer_intent,device,'listener_utterance'))
        exchange_df[data_processer.intent_labels] = pd.DataFrame(exchange_df.empathetic_intent.tolist(),index = exchange_df.index)
        exchange_df = exchange_df.drop(columns=['empathetic_intent','utterance','is_response'])
    if  flag_array[6] == 1:
        if(flag_array[3] == 1):
            #get the emotional similarity, if it is more than 0.7 set mimicry to 1
            exchange_df['emotional_similarity'] = exchange_df.apply(data_processer.get_cosine_similarity,axis = 1) 
            exchange_df['mimicry'] = exchange_df.apply(lambda x: 1 if x['emotional_similarity'] > 0.7 else 0, axis = 1)
            exchange_df = exchange_df.drop(columns = ['emotional_similarity'])
        else: 
            #get emotional mimicry by first computing VAD vectors, since these are not used by the classifier, they are subsequently deleted. 
            exchange_df['vad_speaker'] = exchange_df['speaker_utterance'].apply(lexicon.get_avg_vad, args = (lexicon_df_m,wnl_m,stp_wrds_m)) 
            exchange_df['vad_listener'] = exchange_df['listener_utterance'].apply(lexicon.get_avg_vad, args = (lexicon_df_m,wnl_m,stp_wrds_m)) 
            exchange_df[['valence_speaker','arousal_speaker','dominance_speaker']] = pd.DataFrame(exchange_df.vad_speaker.tolist(),index = exchange_df.index)
            exchange_df[['valence_listener','arousal_listener','dominance_listener']] = pd.DataFrame(exchange_df.vad_listener.tolist(),index = exchange_df.index)
            exchange_df = exchange_df.drop(columns = ['vad_speaker','vad_listener'])                
            exchange_df['emotional_similarity'] = exchange_df.apply(data_processer.get_cosine_similarity,axis = 1) 
            exchange_df['mimicry'] = exchange_df.apply(lambda x: 1 if x['emotional_similarity'] > 0.7 else 0, axis = 1)
            exchange_df = exchange_df.drop(columns =  ['valence_speaker','arousal_speaker','dominance_speaker','valence_listener','arousal_listener','dominance_listener','emotional_similarity'])
        exchange_df['mimicry'] = exchange_df['mimicry'].astype('category')
        exchange_df['mimicry'] = exchange_df['mimicry'].astype('string')
    if flag_array[7] == 1:
        #get emotion labels
        exchange_df['speaker_emotion'] = exchange_df.apply(data_processer.get_emotion_label,axis = 1, args = (emo32_model,emo32_tokenzr,'speaker_utterance')) 
        exchange_df['listener_emotion'] = exchange_df.apply(data_processer.get_emotion_label,axis = 1, args = (emo32_model,emo32_tokenzr,'listener_utterance')) 
        exchange_df = em_red.reduce_emotion_labels_to_8('speaker_emotion',exchange_df)
        exchange_df = em_red.reduce_emotion_labels_to_8('listener_emotion',exchange_df)

    exchange_df = exchange_df[att_lst] #Feature order must match! 
    #Predict empathy
    empathy_prediction = classifier.predict(exchange_df) 

    #output exchange dataframe and prediction
    return exchange_df, empathy_prediction[-1] + 1


def predict_exchange_empathy(classifier, flag_array,ex_num, att_lst, speaker_utterance, listener_utterance, model_components):
    
    sent_model = model_components[0]
    sent_tokenzr = model_components[1]
    epitome_empathy_classifier = model_components[2]
    lexicon_df = model_components[3]
    wnl = model_components[4]
    stp_wrds = model_components[5]
    model_intent = model_components[6]
    tokenizer_intent = model_components[7]
    device = model_components[8]
    lexicon_df_m = model_components[9]
    wnl_m = model_components[10]
    stp_wrds_m = model_components[11]
    emo32_model = model_components[12]
    emo32_tokenzr = model_components[13]
    
   #print(att_lst)
    dummy_dic = {feature: [0] for feature in att_lst}
    #for i in range(len(att_lst)):
    #    #print(att_lst[i])
    #    dummy_dic.update({att_lst[i]: [0]})
    
    exchange_df = pd.DataFrame.from_dict(dummy_dic)
    exchange_df['speaker_utterance'] = speaker_utterance
    exchange_df['listener_utterance'] = listener_utterance

    #Add the features to the exchange dataframe according to the ones used by model. We use the same functions as the experiments, which is why we do everything through dataframes
    if flag_array[0] == 1: 
        #get sentiment
        exchange_df['speaker_sentiment'] = exchange_df.apply(data_processer.get_sentiment_probabilities,axis = 1, args = (sent_model,sent_tokenzr,'speaker_utterance')) 
        exchange_df[['s_negative','s_neutral', 's_positive']] = pd.DataFrame(exchange_df.speaker_sentiment.tolist(),index = exchange_df.index)
        exchange_df['listener_sentiment'] = exchange_df.apply(data_processer.get_sentiment_probabilities,axis = 1, args = (sent_model,sent_tokenzr,'listener_utterance')) 
        exchange_df[['l_negative','l_neutral', 'l_positive']] = pd.DataFrame(exchange_df.listener_sentiment.tolist(),index = exchange_df.index)
        exchange_df = exchange_df.drop(columns=['speaker_sentiment','listener_sentiment'])
    if flag_array[1] == 1:
        #set exchange number to the one we provide
        exchange_df['exchange_number'] = ex_num
    if flag_array[2] == 1:
        #get epitome communication mechanisms
        exchange_df = epitome.classify_epitome_values(epitome_empathy_classifier, exchange_df)
    if flag_array[3] == 1:
        #get VAD vectors
        exchange_df['vad_speaker'] = exchange_df['speaker_utterance'].apply(lexicon.get_avg_vad, args = (lexicon_df,wnl,stp_wrds)) 
        exchange_df['vad_listener'] = exchange_df['listener_utterance'].apply(lexicon.get_avg_vad, args = (lexicon_df,wnl,stp_wrds)) 
        exchange_df[['valence_speaker','arousal_speaker','dominance_speaker']] = pd.DataFrame(exchange_df.vad_speaker.tolist(),index = exchange_df.index)
        exchange_df[['valence_listener','arousal_listener','dominance_listener']] = pd.DataFrame(exchange_df.vad_listener.tolist(),index = exchange_df.index)
        exchange_df = exchange_df.drop(columns = ['vad_speaker','vad_listener'])
    if flag_array[4] == 1:
        #get word lengths
        exchange_df['s_word_len'] = exchange_df['speaker_utterance'].apply(data_processer.get_word_len) 
        exchange_df['l_word_len'] = exchange_df['listener_utterance'].apply(data_processer.get_word_len) 
    if flag_array[5] == 1: 
        #get intent
        exchange_df['utterance'] = exchange_df['listener_utterance']
        exchange_df['is_response'] = 1
        exchange_df['empathetic_intent'] = exchange_df.apply(data_processer.get_emp_intent_probabilities, axis=1, args = (model_intent,tokenizer_intent,device,'listener_utterance'))
        exchange_df[data_processer.intent_labels] = pd.DataFrame(exchange_df.empathetic_intent.tolist(),index = exchange_df.index)
        exchange_df = exchange_df.drop(columns=['empathetic_intent','utterance','is_response'])
    if  flag_array[6] == 1:
        if(flag_array[3] == 1):
            #get the emotional similarity, if it is more than 0.7 set mimicry to 1
            exchange_df['emotional_similarity'] = exchange_df.apply(data_processer.get_cosine_similarity,axis = 1) 
            exchange_df['mimicry'] = exchange_df.apply(lambda x: 1 if x['emotional_similarity'] > 0.7 else 0, axis = 1)
            exchange_df = exchange_df.drop(columns = ['emotional_similarity'])
        else: 
            #get emotional mimicry by first computing VAD vectors, since these are not used by the classifier, they are subsequently deleted. 
            exchange_df['vad_speaker'] = exchange_df['speaker_utterance'].apply(lexicon.get_avg_vad, args = (lexicon_df_m,wnl_m,stp_wrds_m)) 
            exchange_df['vad_listener'] = exchange_df['listener_utterance'].apply(lexicon.get_avg_vad, args = (lexicon_df_m,wnl_m,stp_wrds_m)) 
            exchange_df[['valence_speaker','arousal_speaker','dominance_speaker']] = pd.DataFrame(exchange_df.vad_speaker.tolist(),index = exchange_df.index)
            exchange_df[['valence_listener','arousal_listener','dominance_listener']] = pd.DataFrame(exchange_df.vad_listener.tolist(),index = exchange_df.index)
            exchange_df = exchange_df.drop(columns = ['vad_speaker','vad_listener'])                
            exchange_df['emotional_similarity'] = exchange_df.apply(data_processer.get_cosine_similarity,axis = 1) 
            exchange_df['mimicry'] = exchange_df.apply(lambda x: 1 if x['emotional_similarity'] > 0.7 else 0, axis = 1)
            exchange_df = exchange_df.drop(columns =  ['valence_speaker','arousal_speaker','dominance_speaker','valence_listener','arousal_listener','dominance_listener','emotional_similarity'])
        exchange_df['mimicry'] = exchange_df['mimicry'].astype('category')
        exchange_df['mimicry'] = exchange_df['mimicry'].astype('string')
    if flag_array[7] == 1:
        #get emotion labels
        exchange_df['speaker_emotion'] = exchange_df.apply(data_processer.get_emotion_label,axis = 1, args = (emo32_model,emo32_tokenzr,'speaker_utterance')) 
        exchange_df['listener_emotion'] = exchange_df.apply(data_processer.get_emotion_label,axis = 1, args = (emo32_model,emo32_tokenzr,'listener_utterance')) 
        exchange_df = em_red.reduce_emotion_labels_to_8('speaker_emotion',exchange_df)
        exchange_df = em_red.reduce_emotion_labels_to_8('listener_emotion',exchange_df)

    exchange_df = exchange_df[att_lst] #Feature order must match! 
    #Predict empathy
    empathy_prediction = classifier.predict(exchange_df) 

    #output exchange dataframe and prediction
    return exchange_df, empathy_prediction[-1] + 1




def fix_mimicry(list):
    #This just changes the mimicry fature to have a 'binary' descriptor instead of a list to make it hashable
    if(('mimicry', ['0', '1']) in list):
        mimicry_index = list.index(('mimicry', ['0', '1']))
        list[mimicry_index] = ('mimicry', 'binary')
    return list

def is_greater_than(feature_name, pattern_items):
    #We use this function to get if the feature in a pattern implies an increase of that feature
    for item in pattern_items:
        if feature_name in str(item) and '>' in str(item):
            return True
    return False

def recursive_search(search_control, intersections,items,role):
    #print('recursive')
    
    #start the search
    if search_control == 0:
        if len(intersections[0]) > 0:
            return recursive_search(1,intersections,items,role)
        elif len(intersections[1]) > 0:
            return recursive_search(2,intersections,items,role)
        elif len(intersections[2]) > 0:
            return recursive_search(3,intersections,items,role)            
        elif len(intersections[3]) > 0:
            return recursive_search(4,intersections,items,role)
        elif len(intersections[4]) > 0:
            return recursive_search(5,intersections,items,role)
        elif len(intersections[5]) > 0:
            return recursive_search(6,intersections,items,role)
        else:
            #print('NO FEATURES AVAILABLE')
            return 'ERROR: No features available for this purpose'
    #The gist of this function is that it searches for features in the most representative pattern of a class of empathy. If it finds it and the pattern says that that feature is higher (>), we make a recommendation to increase that feature in the response
    if search_control == 1:
        #print('VAD')
        if ('valence_listener' in intersections[0]) and (is_greater_than('valence_listener', items)):
            #return 'Maybe we should be more positive!'
            return "Suggestion: Increase valence"
        elif 'arousal_listener' in intersections[0] and (is_greater_than('arousal_listener', items)):
            #return 'How about we say our feelings with more intensity!'
            #return "Suggestion: Increase arousal"
            return "Suggestion: Maybe we should focus on the feelings of the other person!" 
        elif 'arousal_listener' in intersections[0] and (is_greater_than('arousal_listener', items)):
            #return "Remember "+ role +" confidence is important!"
            return "Suggestion: "+ role +" should display more confidence (dominance)"
        else:
            #print('NO VAD')
            return recursive_search(search_control+1, intersections,items,role)
    elif search_control == 2:
        #print('Intent')
        if ('questioning' in intersections[1]) and (is_greater_than('questioning', items)):
            #return "Hey "+role+" why not ask them a question? I also want to know more"
            #return "Suggestion: " + str(role) + ", I think you should ask them a question about what they say."
            return "Great chat. Though I am still curious about some things, maybe next time you could ask eachother more questions "
        elif ('acknowledging' in intersections[1]) and (is_greater_than('acknowledging', items)):
            #print("I think you it is important to say that you have a right to feel the way you feel, whether it is good or bad. Even though we disagree sometimes!")
            return "I think you it is important to say that you have a right to feel the way you feel, whether it is good or bad. Even though we disagree sometimes!"
            #print(items)
        #    if ('s_positive <' in str(items)) or ('s_negative >' in str(items)) or ('valence_speaker <' in str(items)):
        #        return "Suggestion: Acknowledge their negative feelings, " + role
        #    else:
        #        return "Suggestion: Acknowledge their positive feelings, " + role
        elif ('agreeing' in intersections[1]) and (is_greater_than('agreeing', items)):
            return "Suggestion: I think they were looking for agreement "+role+"."
            #return "I think they were looking for agreement "+role+"."
        elif ('encouraging' in intersections[1]) and (is_greater_than('encouraging', items)):
            return "Suggestion: My sensors say that we should encourage them about their feelings "+role+""
            #return "My sensors say that we should encourage them about their feelings "+role+"?"
        elif ('consoling' in intersections[1]) and (is_greater_than('consoling', items)):
            return "Suggestion: Consolation should be provided"
            #return "Oh no "+role+" Maybe we could say something to cheer them up!"       
        elif ('suggesting' in intersections[1]) and (is_greater_than('suggesting', items)):
            return "Suggestion: Suggest something they could do"
            #return 'What do you think they should do?'
        elif ('wishing' in intersections[1]) and (is_greater_than('wishing', items)):
            return "Suggestion: Humans wish eachother good things in these situations"
            #return "I hear humans wish eachother good things in these situations "+role+". Why don't you give it a try?"
        elif ('sympathizing' in intersections[1]) and (is_greater_than('sympathizing', items)):
            return "Suggestion: Respond with sympathy"
            #return 'Oh, I feel bad for them. Do you feel like that too?'  
        else:
            return recursive_search(search_control+1, intersections,items,role)
    elif search_control == 3:
        #print('Sentiment')
        if ('l_positive' in intersections[1]) and (is_greater_than('l_positive', items)):
            return "Suggestion: Be more positive"
            #return 'Hmm, maybe we should be more positive'
        elif ('l_negative' in intersections[2]) and (is_greater_than('l_negative', items)):
            #return "Oh, that's bad. Don't you agree "+role+"?"
            return "Suggestion: Acknowledge their negative feelings"
        else:
            return recursive_search(search_control+1, intersections,items,role)
    elif search_control == 4:
        #print('epitome')
        if ('predictions_ER' in intersections[3]) and (is_greater_than('predictions_ER', items)):
            return 'How about we say our feelings with more intensity!'
        elif ('predictions_EX' in intersections[3]) and (is_greater_than('predictions_EX', items)):
            return "Hey "+role+" why not ask them a question? I also want to know more"
        elif ('predictions_IP' in intersections[3]) and (is_greater_than('predictions_IP', items)):
            return "Do you understand what they mean?"   
        else:
            return recursive_search(search_control+1, intersections,items,role)
    elif search_control == 5:
        if len(intersections[4]) > 0:
            return 'Do you feel the same way?'
        else:
            return recursive_search(search_control+1, intersections,items,role)
    elif search_control == 6:
        if len(intersections[5]) > 0:
            return 'Maybe you should elaborate more'
        else:
            return recursive_search(search_control+1, intersections,items,role)
    else:
        #print('NO FEATURES AVAILABLE')
        return 'ERROR'


def compute_feature_intersections(feature_names):
    #Features from the 'listener' role
    intent_features = ['agreeing', 'acknowledging', 'encouraging', 'consoling', 'sympathizing', 'suggesting', 'questioning', 'wishing']
    sentiment_features = ['l_negative', 'l_neutral', 'l_positive']
    epitome_features = ['predictions_ER', 'predictions_EX', 'predictions_IP']
    length_features = ['l_word_len']
    vad_features = ['valence_listener', 'arousal_listener', 'dominance_listener']
    mimicry_features = ['mimicry']

    intent_intersection = set(intent_features).intersection(set(feature_names))
    vad_intersection = set(vad_features).intersection(set(feature_names))
    sentiment_intersection = set(sentiment_features).intersection(set(feature_names))
    epitome_intersection = set(sentiment_features).intersection(set(feature_names))
    mimicry_intersection = set(mimicry_features).intersection(set(feature_names))
    len_intersection = set(length_features).intersection(set(feature_names))

    intersection_arr = [vad_intersection,intent_intersection,sentiment_intersection,epitome_intersection,mimicry_intersection,len_intersection]
    return intersection_arr
    


def get_recommentation(classifier, exchange_df,role):
    #get most influential patterns
    
    emerging_patterns = classifier.EmergingPatterns #access the patterns mined by the classifier
    pattern_list = [] #patterns that cover the exchange
    for instance in exchange_df.to_numpy(): 
        for pattern in emerging_patterns:
            if pattern.IsMatch(instance):
                pattern_list.append(pattern)   
    count_lst = [pattern.Counts for pattern in pattern_list]
    count_arr = np.array(count_lst)
    most_support_low = pattern_list[count_arr[:,0].argmax()] #Pattern that most supports the lowest class
    most_support_mid = pattern_list[count_arr[:,1].argmax()] #Pattern that most supports the middle class
    most_support_high = pattern_list[count_arr[:,2].argmax()] #Pattern that most supports the high class

    #print(most_support_high)
    
    most_support_high_items = most_support_high.Items
    most_support_mid_items = most_support_mid.Items
    most_support_low_items = most_support_low.Items
    
    #get the features from most supported patterns
    most_support_high_features = [items.Feature for items in most_support_high.Items]
    most_support_mid_features = [items.Feature for items in most_support_mid.Items]
    most_support_low_features =[items.Feature for items in most_support_low.Items]

    most_support_high_features = fix_mimicry(most_support_high_features)
    most_support_mid_features = fix_mimicry(most_support_mid_features)
    most_support_low_features = fix_mimicry(most_support_low_features)


    #Obtain the features that differenciate the low features from the others
    features_only_on_low = set(most_support_low_features) - set(most_support_high_features) - set(most_support_mid_features) 
    features_only_on_mid = set(most_support_mid_features) - set(most_support_low_features)
    features_only_on_high = set(most_support_high_features) - set(most_support_low_features)

    #print(most_support_high)
    #print(features_only_on_high)
    
    #get the names of the features on the class of interest
    feature_names_high = [features[0] for features in features_only_on_high]
    feature_names_mid = [features[0] for features in features_only_on_mid]
    fearure_names_low = [features[0] for features in features_only_on_low]

    #First, we try to improve empathy using the features from the pattern most representative of the high empathy class
    intersections = compute_feature_intersections(feature_names_high)
    response = recursive_search(0,intersections,most_support_high_items, str(role))
    #print(type(response))
    if 'ERROR' in response:
        #if that fails, we try to get a response using the features from the middle empathy class
        intersections = compute_feature_intersections(feature_names_mid)
        response = recursive_search(0,intersections,most_support_mid_items,str(role))
        if 'ERROR' in response:
            #If that fails, we just give up (maybe we should try to do the reverse using the features on the low empathy class i.e. check if there are features we could reduce (<))
            print('Maybe we should talk about something else')
    
    #print(response)
    return response






def get_most_relevant_feature(classifier, exchange_df,role):
    #get most influential patterns
    
    emerging_patterns = classifier.EmergingPatterns #access the patterns mined by the classifier
    pattern_list = [] #patterns that cover the exchange
    for instance in exchange_df.to_numpy(): 
        for pattern in emerging_patterns:
            if pattern.IsMatch(instance):
                pattern_list.append(pattern)   
    count_lst = [pattern.Counts for pattern in pattern_list]
    count_arr = np.array(count_lst)
    print(count_arr)

    most_support_low = pattern_list[count_arr[:,0].argmax()] #Pattern that most supports the lowest class
    most_support_mid = pattern_list[count_arr[:,1].argmax()] #Pattern that most supports the middle class
    most_support_high = pattern_list[count_arr[:,2].argmax()] #Pattern that most supports the high class

    #print(most_support_high)
    
    most_support_high_items = most_support_high.Items
    most_support_mid_items = most_support_mid.Items
    most_support_low_items = most_support_low.Items
    
    #get the features from most supported patterns
    most_support_high_features = [items.Feature for items in most_support_high.Items]
    most_support_mid_features = [items.Feature for items in most_support_mid.Items]
    most_support_low_features =[items.Feature for items in most_support_low.Items]

    most_support_high_features = fix_mimicry(most_support_high_features)
    most_support_mid_features = fix_mimicry(most_support_mid_features)
    most_support_low_features = fix_mimicry(most_support_low_features)


    #Obtain the features that differenciate the low features from the others
    features_only_on_low = set(most_support_low_features) - set(most_support_high_features) - set(most_support_mid_features) 
    features_only_on_mid = set(most_support_mid_features) - set(most_support_low_features)
    features_only_on_high = set(most_support_high_features) - set(most_support_low_features)

    #print(most_support_high)
    #print(features_only_on_high)
    
    #get the names of the features on the class of interest
    feature_names_high = [features[0] for features in features_only_on_high]
    feature_names_mid = [features[0] for features in features_only_on_mid]
    fearure_names_low = [features[0] for features in features_only_on_low]

    #First, we try to improve empathy using the features from the pattern most representative of the high empathy class
    intersections = compute_feature_intersections(feature_names_high)
    response = recursive_search(0,intersections,most_support_high_items, str(role))
    #print(type(response))
    if 'ERROR' in response:
        #if that fails, we try to get a response using the features from the middle empathy class
        intersections = compute_feature_intersections(feature_names_mid)
        response = recursive_search(0,intersections,most_support_mid_items,str(role))
        if 'ERROR' in response:
            #If that fails, we just give up (maybe we should try to do the reverse using the features on the low empathy class i.e. check if there are features we could reduce (<))
            print('Maybe we should talk about something else')
    
    #print(response)
    return response



