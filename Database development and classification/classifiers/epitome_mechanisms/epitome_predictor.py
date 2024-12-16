import csv
import pandas as pd
import torch
from .src.empathy_classifier import EmpathyClassifier



'''
Example:
'''

def load_epitome_classifier(mdl_path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    empathy_classifier = EmpathyClassifier(device,
                        ER_model_path = mdl_path+'/reddit_ER.pth', 
                        IP_model_path = mdl_path+'/reddit_IP.pth',
                        EX_model_path = mdl_path+'/reddit_EX.pth')


    return empathy_classifier

def classify_epitome_values(empathy_classifier,in_df):
    input_df = in_df
    epitome_df = input_df.copy()
    
    epitome_df['predictions_ER'] = 0
    epitome_df['predictions_IP'] = 0
    epitome_df['predictions_EX'] = 0
    for i in range(len(epitome_df['speaker_utterance'])):
    #(logits_empathy_ER, predictions_ER, logits_empathy_IP, predictions_IP, logits_empathy_EX, predictions_EX, logits_rationale_ER, predictions_rationale_ER, logits_rationale_IP, predictions_rationale_IP, logits_rationale_EX,predictions_rationale_EX) = empathy_classifier.predict_empathy([epitome_df.loc[i, 'seeker_post']], [epitome_df.loc[i, 'response_post']])
        (logits_empathy_ER, predictions_ER, logits_empathy_IP, predictions_IP, logits_empathy_EX, predictions_EX, logits_rationale_ER, predictions_rationale_ER, logits_rationale_IP, predictions_rationale_IP, logits_rationale_EX,predictions_rationale_EX) = empathy_classifier.predict_empathy([str(epitome_df.loc[i, 'speaker_utterance'])], [str(epitome_df.loc[i, 'listener_utterance'])])
        #epitome_df.loc[i] = [ids[i], seeker_posts[i], response_posts[i], predictions_ER[0], predictions_IP[0], predictions_EX[0], predictions_rationale_ER[0].tolist(), predictions_rationale_IP[0].tolist(), predictions_rationale_EX[0].tolist()]
        epitome_df.loc[i, 'predictions_ER'] = predictions_ER[0]
        epitome_df.loc[i, 'predictions_IP'] = predictions_IP[0]
        epitome_df.loc[i, 'predictions_EX'] = predictions_EX[0]
    #print(epitome_df)
    return epitome_df   

def predict_epitome_values(mdl_path,in_df):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    input_df = in_df


    empathy_classifier = EmpathyClassifier(device,
							ER_model_path = mdl_path+'/reddit_ER.pth', 
							IP_model_path = mdl_path+'/reddit_IP.pth',
							EX_model_path = mdl_path+'/reddit_EX.pth')

    epitome_df = input_df.copy()
    
    epitome_df['predictions_ER'] = 0
    epitome_df['predictions_IP'] = 0
    epitome_df['predictions_EX'] = 0
    #epitome_df = pd.DataFrame(columns=['ids', 'seeker_posts', 'response_posts', 'predictions_ER', 'predictions_IP', 'predictions_EX', 'predictions_rationale_ER', 'predictions_rationale_IP', 'predictions_rationale_EX'])
    for i in range(len(epitome_df['speaker_utterance'])):
        #(logits_empathy_ER, predictions_ER, logits_empathy_IP, predictions_IP, logits_empathy_EX, predictions_EX, logits_rationale_ER, predictions_rationale_ER, logits_rationale_IP, predictions_rationale_IP, logits_rationale_EX,predictions_rationale_EX) = empathy_classifier.predict_empathy([epitome_df.loc[i, 'seeker_post']], [epitome_df.loc[i, 'response_post']])
        (logits_empathy_ER, predictions_ER, logits_empathy_IP, predictions_IP, logits_empathy_EX, predictions_EX, logits_rationale_ER, predictions_rationale_ER, logits_rationale_IP, predictions_rationale_IP, logits_rationale_EX,predictions_rationale_EX) = empathy_classifier.predict_empathy([str(epitome_df.loc[i, 'speaker_utterance'])], [str(epitome_df.loc[i, 'listener_utterance'])])
        #epitome_df.loc[i] = [ids[i], seeker_posts[i], response_posts[i], predictions_ER[0], predictions_IP[0], predictions_EX[0], predictions_rationale_ER[0].tolist(), predictions_rationale_IP[0].tolist(), predictions_rationale_EX[0].tolist()]
        epitome_df.loc[i, 'predictions_ER'] = predictions_ER[0]
        epitome_df.loc[i, 'predictions_IP'] = predictions_IP[0]
        epitome_df.loc[i, 'predictions_EX'] = predictions_EX[0]
    #print(epitome_df)
    return epitome_df


'''
csv_writer = csv.writer(output_file, delimiter=',', quotechar='"')

csv_writer.writerow(['id','seeker_post','response_post','ER_label','IP_label','EX_label', 'ER_rationale', 'IP_rationale', 'EX_rationale'])

for i in range(len(seeker_posts)):
	(logits_empathy_ER, predictions_ER, logits_empathy_IP, predictions_IP, logits_empathy_EX, predictions_EX, logits_rationale_ER, predictions_rationale_ER, logits_rationale_IP, predictions_rationale_IP, logits_rationale_EX,predictions_rationale_EX) = empathy_classifier.predict_empathy([seeker_posts[i]], [response_posts[i]])

	csv_writer.writerow([ids[i], seeker_posts[i], response_posts[i], predictions_ER[0], predictions_IP[0], predictions_EX[0], predictions_rationale_ER[0].tolist(), predictions_rationale_IP[0].tolist(), predictions_rationale_EX[0].tolist()])

output_file.close()
'''

#predict_epitome_values()

def ping():
    
    print('ping!')
    #print(dir(tst))
    return 0


def setup():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device
