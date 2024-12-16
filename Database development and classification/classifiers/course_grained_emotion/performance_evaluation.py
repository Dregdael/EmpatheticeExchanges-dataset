
import os
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix, top_k_accuracy_score,balanced_accuracy_score
import pandas as pd
import emotion_reductor as em_red

current_dir = os.path.dirname(os.path.abspath(__file__))

def encode_emotion(emotion_column,dataframe):
    dataframe[str(emotion_column)] = dataframe[str(emotion_column)].astype('category')
    dataframe[emotion_column + "_encoded"] = dataframe[emotion_column].cat.codes
    c = dataframe[emotion_column].astype('category')
    print(dict(enumerate(c.cat.categories)))
    return dataframe

df = pd.read_csv('/home/haru/EmpathyClassification_ECPC/processed_databases/EmpatheticExchanges/EmpatheticExchanges.csv')

#emotion reduction
df = em_red.reduce_emotion_labels_to_8('context',df)
df = em_red.reduce_emotion_labels_to_8('speaker_emotion',df)

#df = em_red.reduce_emotion_labels('context',df)
#df = em_red.reduce_emotion_labels('speaker_emotion',df)

#Encoding for evaluation
df = encode_emotion('context',df)

df = encode_emotion('speaker_emotion',df)

y = df.context_encoded
y_pred = df.speaker_emotion_encoded

acc = accuracy_score(y, y_pred)
conf_mat = confusion_matrix(y,y_pred)
adjusted_acc = balanced_accuracy_score(y, y_pred, adjusted = True)


print(df.head())
print(acc)
print(conf_mat)
print(adjusted_acc)