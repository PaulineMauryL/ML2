import numpy as np
import pandas as pd


def get_ids(data):
    '''Get ids from testing set'''
    with open(data, "r") as file:
        ids = []
        for _, line in enumerate(file):
            ids.append( line.split(',', 1)[0] )
    return ids

def read_data(data):
    with open(data, "r",encoding="latin1") as file:
        tweets = str()
        for _,line in enumerate(file):
            tweets += line
        tweets = tweets.split('\n')
        del tweets[-1]
    return tweets

def get_data(pos = "./datasets/train_pos.txt", neg = "./datasets/train_neg.txt"):
    X_pos = read_data(pos)
    X_neg = read_data(neg)
    X_train = X_pos + X_neg
    
    Y_pos = np.ones(len(X_pos), dtype = int)
    Y_neg = np.zeros(len(X_neg), dtype = int)
    Y_train = np.concatenate((Y_pos, Y_neg), axis = -1)
    
    X_train_shuffled, Y_train_shuffled = shuffle(X_train, Y_train, random_state=52)
    
    return X_train_shuffled, Y_train_shuffled

def read_glove_vecs_only_alpha(glove_file):
    with open(glove_file, 'r',encoding='utf8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            if curr_word.isalpha() or curr_word.startswith("#"):
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def create_csv_submission(ids, y_pred, name):
    '''
    Function taken from helpers of project 1: Creates an output file in csv format for submission to CrowdAI
    Input: 
    ids (event ids associated with each prediction)
    y_pred (predicted class labels)
    name (string name of .csv output file to be created)
    '''
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
