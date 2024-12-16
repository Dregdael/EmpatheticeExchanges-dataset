import numpy as np
import pandas as pd
import math


def get_difference_between_classes(class1,class2,counts):
    sum = 0
    if class1<=class2:
        for i in range(class1+1,class2+1):
            sum += counts[i]
    else:
        for i in range(class1-1,class2-1,-1):
            sum += counts[i]
    return sum 

def get_class_counts(y):
    class_count = y.value_counts()
    return class_count

def get_proximity_matrix(y):
    y['empathy'] = y["empathy"].astype('int')
    n_per_class = get_class_counts(y)
    num_of_classes = len(n_per_class)
    n_total = len(y)

    proximity_matrix = [[0]* num_of_classes for i in range(num_of_classes)]

    for i in range(num_of_classes):
        for j in range(num_of_classes):
            class_diff = get_difference_between_classes(i+1,j+1,n_per_class)
            proximity_matrix[i][j] = -math.log(((n_per_class[i+1]/2) + class_diff)/n_total,2) 

    #for i in range(num_of_classes):
    #    for j in range(num_of_classes):
    #        print(f'{proximity_matrix[i][j]:.2f}', end=' ')
    #    print()
    
    return proximity_matrix

def get_numerator(confusion,proximity):
    numerator = 0
    row_sum = 0
    for i in range(len(confusion)):
        for j in range(len(confusion[0])): 
            row_sum += confusion[i][j]*proximity[i][j]
        numerator += row_sum
    #    print(f'row sum: {row_sum:.2f}, numerator accumulated: {numerator:.2f}')
        row_sum = 0
    return numerator

def get_denominator(y,proximity):
    denominator = 0
    n_per_class = get_class_counts(y)
    #print(n_per_class)
    num_of_classes = len(n_per_class)
    for i in range(num_of_classes):
        denominator += n_per_class[i+1]*proximity[i][i]
    return denominator

def get_cem(y_predicted,y):
    proximity = get_proximity_matrix(y)
    confusion = get_confusion_matrix(y_predicted,y)
    num = get_numerator(confusion,proximity)
    den = get_denominator(y,proximity)
    cem = num/den
    return cem



def get_confusion_matrix(y_predicted, y):
    y = y['empathy']
    n_per_class = get_class_counts(y)
    num_of_classes = len(n_per_class)
    confusion_matrix = [[0]* num_of_classes for i in range(num_of_classes)]
    
    good_guess = 0
    for i in range(len(y)):
        confusion_matrix[y_predicted[i]][y[i]-1] += 1
    #for i in range(num_of_classes):
    #    for j in range(num_of_classes):
    #        print(f'{confusion_matrix[i][j]}', end=' ')
    #    print()
    return confusion_matrix



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

