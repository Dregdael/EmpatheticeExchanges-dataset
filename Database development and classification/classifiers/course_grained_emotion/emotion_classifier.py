from datasets import load_dataset
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

dataset = load_dataset("bdotloh/empathetic-dialogues-contexts")

print(dataset)

df_train = dataset['train'].to_pandas()

#df_train = df_train[0:10]

print(df_train.head())

df_train["emotion"] = df_train["emotion"].astype('category')

df_train["emotion_encoded"] = df_train["emotion"].cat.codes

c = df_train.emotion.astype('category')
d = dict(enumerate(c.cat.categories))


print(d)

print(df_train.head())


#df["context_encoded"] = df["context_encoded"].astype('category')

plutchik_emotions = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]

plutchik_equivalencies = [[[0,0,1,0,0,0,0,0],2], #afraid
                          [[0,0,0,0,0,0,1,0],2], #angry
                          [[0,0,0,0,0,0,1,0],3], #annoyed
                          [[0,0,0,0,0,0,0,1],2], #anticipating
                          [[0,0,1,0,0,0,0,0],2], #anxious
                          [[0,0,1,0,0,0,0,0],3], #apprehensive
                          [[0,0,1,0,0,1,0,0],2], #ashamed
                          [[1,1,0,0,0,0,0,0],2], #caring
                          [[1,0,0,0,0,0,0,1],2], #confident
                          [[1,0,0,0,0,0,0,0],3], #content
                          [[0,0,0,1,1,1,0,0],1], #devastated
                          [[0,0,0,1,1,0,0,0],2], #disappointed
                          [[0,0,0,0,0,1,0,0],2], #disgusted
                          [[0,0,1,0,0,1,0,0],3], #embarrassed
                          [[1,0,0,0,0,0,0,1],3], #excited
                          [[1,1,0,1,0,0,0,0],1], #faithful
                          [[0,0,0,0,0,0,1,0],1], #furious
                          [[1,1,0,1,0,0,0,0],2], #grateful
                          [[1,0,1,0,0,0,0,0],2], #guilty
                          [[0,1,0,0,0,0,0,1],2], #hopeful
                          [[0,1,0,1,0,0,0,0],1], #impressed
                          [[0,0,0,0,1,0,1,0],2], #jealous
                          [[1,0,0,0,0,0,0,0],2], #joyful
                          [[0,0,0,0,1,0,0,0],1], #lonely
                          [[1,0,0,0,1,0,0,0],2], #nostalgic
                          [[0,0,0,0,0,0,0,1],2], #prepared
                          [[1,0,0,0,0,0,1,0],2], #proud
                          [[0,0,0,0,1,0,0,0],2], #sad
                          [[0,1,0,0,0,0,0,0],2], #sentimental
                          [[0,0,0,1,0,0,0,0],2], #surprised
                          [[0,0,1,0,0,0,0,0],1], #terrified
                          [[0,1,0,0,0,0,0,0],2]] #trusting

plutchik_equivalencies_wo_intensity = []
for i in range(len(plutchik_equivalencies)):
    plutchik_equivalencies_wo_intensity.append(plutchik_equivalencies[i][0])

#print(plutchik_equivalencies_wo_intensity)

def get_reduced_label(vector):
    if vector.count(1) > 2:
        return d[plutchik_equivalencies_wo_intensity.index(vector)]
    elif vector.count(1) == 2:
        return d[plutchik_equivalencies_wo_intensity.index(vector)]
    else:
        return plutchik_emotions[vector.index(1)]
    


df_train['emotion_plutchik'] = df_train['emotion_encoded'].apply(lambda x: plutchik_equivalencies[x][0])


df_train['plutchik_labels'] = df_train['emotion_plutchik'].apply(get_reduced_label)



hist = df_train['plutchik_labels'].hist()
plt.show() 

print(df_train['plutchik_labels'].unique())

print(df_train.head())