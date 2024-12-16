
def get_reduced_label(vector,plutchik_8_vector,list_8_emotions,d):
    print(vector)
    if vector[0].count(1) > 2:
        print('>2')
        print(plutchik_8_vector.index(vector))
        return d[plutchik_8_vector.index(vector)]
    elif vector[0].count(1) == 2:
        print('2')
        print(plutchik_8_vector.index(vector))
        return d[plutchik_8_vector.index(vector)]
    else:
        print(list_8_emotions[vector[0].index(1)])
        return list_8_emotions[vector[0].index(1)]




def reduce_emotion_labels(emotion_column,dataframe):
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
                            [[0,0,1,0,0,1,0,0],3], #embarassed
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

    dictionary = {'afraid':'fear',
                  'angry': 'anger',
                  'annoyed': 'anger',
                  'anticipating': 'anticipation',
                  'anxious': 'fear',
                  'apprehensive': 'fear',
                  'ashamed': 'ashamed',
                  'caring': 'caring',
                  'confident': 'confident',
                  'content': 'joy',
                  'devastated': 'devastated',
                  'disappointed': 'disappointed',
                  'disgusted':'disgust',
                  'embarrassed':'embarassed',
                  'excited':'excited',
                  'faithful':'faithful',
                  'furious':'anger',
                  'grateful':'grateful',
                  'guilty':'guilty',
                  'hopeful':'hopeful',
                  'impressed':'impressed',
                  'jealous':'jealous',
                  'joyful':'joy',
                  'lonely':'sadness',
                  'nostalgic':'nostalgic',
                  'prepared':'anticipation',
                  'proud':'proud',
                  'sad':'sadness',
                  'sentimental':'trust',
                  'surprised':'surprise',
                  'terrified':'fear',
                  'trusting':'trust',
                  'joy': 'joy',
                  'trust': 'trust',
                  'fear': 'fear',
                  'surprise': 'surprise',
                  'sadness': 'sadness',
                  'disgust': 'disgust',
                  'anger': 'anger',
                  'anticipation': 'anticipation'
                  }
    
    dataframe[emotion_column] = dataframe[emotion_column].apply(lambda x: dictionary[x])


    return dataframe




def reduce_emotion_labels_to_8(emotion_column,dataframe):

    plutchik_emotions = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]

    plutchik_equivalencies = [[[0,0,1,0,0,0,0,0],2], #afraid
                            [[0,0,0,0,0,0,1,0],2], #angry
                            [[0,0,0,0,0,0,1,0],3], #annoyed
                            [[0,0,0,0,0,0,0,1],2], #anticipating
                            [[0,0,1,0,0,0,0,0],2], #anxious
                            [[0,0,1,0,0,0,0,0],3], #apprehensive
                            [[0,0,0,0,0,1,0,0],2], #ashamed
                            [[0,1,0,0,0,0,0,0],2], #caring
                            [[0,0,0,0,0,0,0,1],2], #confident
                            [[1,0,0,0,0,0,0,0],3], #content
                            [[0,0,0,0,1,0,0,0],1], #devastated
                            [[0,0,0,0,1,0,0,0],2], #disappointed
                            [[0,0,0,0,0,1,0,0],2], #disgusted
                            [[0,0,0,0,0,1,0,0],3], #embarassed
                            [[1,0,0,0,0,0,0,0],3], #excited
                            [[0,1,0,0,0,0,0,0],1], #faithful
                            [[0,0,0,0,0,0,1,0],1], #furious
                            [[0,1,0,0,0,0,0,0],2], #grateful
                            [[0,0,0,0,0,1,0,0],2], #guilty
                            [[0,0,0,0,0,0,0,1],2], #hopeful
                            [[0,0,0,1,0,0,0,0],1], #impressed
                            [[0,0,0,0,1,0,0,0],2], #jealous
                            [[1,0,0,0,0,0,0,0],2], #joyful
                            [[0,0,0,0,1,0,0,0],1], #lonely
                            [[0,0,0,0,1,0,0,0],2], #nostalgic
                            [[0,0,0,0,0,0,0,1],2], #prepared
                            [[1,0,0,0,0,0,0,0],2], #proud
                            [[0,0,0,0,1,0,0,0],2], #sad
                            [[0,0,0,0,1,0,0,0],2], #sentimental
                            [[0,0,0,1,0,0,0,0],2], #surprised
                            [[0,0,1,0,0,0,0,0],1], #terrified
                            [[0,1,0,0,0,0,0,0],2]] #trusting

    dictionary = {'afraid':'fear',
                  'angry': 'anger',
                  'annoyed': 'anger',
                  'anticipating': 'anticipation',
                  'anxious': 'fear',
                  'apprehensive': 'fear',
                  'ashamed': 'disgust',
                  'caring': 'trust',
                  'confident': 'anticipation',
                  'content': 'joy',
                  'devastated': 'sadness',
                  'disappointed': 'sadness',
                  'disgusted':'disgust',
                  'embarrassed':'disgust',
                  'excited':'joy',
                  'faithful':'trust',
                  'furious':'anger',
                  'grateful':'trust',
                  'guilty':'disgust',
                  'hopeful':'anticipation',
                  'impressed':'surprise',
                  'jealous':'sadness',
                  'joyful':'joy',
                  'lonely':'sadness',
                  'nostalgic':'sadness',
                  'prepared':'anticipation',
                  'proud':'joy',
                  'sad':'sadness',
                  'sentimental':'trust',
                  'surprised':'surprise',
                  'terrified':'fear',
                  'trusting':'trust',
                  'joy': 'joy',
                  'trust': 'trust',
                  'fear': 'fear',
                  'surprise': 'surprise',
                  'sadness': 'sadness',
                  'disgust': 'disgust',
                  'anger': 'anger',
                  'anticipation': 'anticipation'
                  }
    
    dataframe[emotion_column] = dataframe[emotion_column].apply(lambda x: dictionary[x])


    return dataframe

def compare(value1,value2):
    if value1 == value2:
        return 1
    else:
        return 0

def get_mimicry(emotion_column1,emotion_column2, dataframe):
    #print(dataframe)
    dataframe['mimicry'] = dataframe.apply(lambda x: 1 if x[emotion_column1] == x[emotion_column2] else 0, axis = 1)
    return dataframe

