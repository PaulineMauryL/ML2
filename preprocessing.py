#===========================================================================================
#===========================================================================================
#============================ PREPROCESSING ================================================
#===========================================================================================
#===========================================================================================

import numpy as np
import pandas as pd
import re
from our_functionsv3 import read_data, get_data, get_test_data, convert_to_one_hot, get_test_data
from sklearn.utils import shuffle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
from nltk.stem.porter import PorterStemmer

def pre_process_tweets(data):
    
    # make a copy to be sure that data itself is not changed and that we can compare it later.
    data2=data.copy()
    
    # change n't into not to keep this information. Without it, the words with n't would just be erased.
    data2["tweet"] = data2["tweet"].str.replace("n't", " not")
    
    # go into preprocessing to separate all words and punctuation
    data2["tweet"] = data2["tweet"].apply(lambda x: preprocess1(x))
    
    # reconstruct the #
    data2["tweet"] = data2["tweet"].str.replace("# ", "#")
    
    # go into preprocessing
    data2["tweet"] = data2["tweet"].apply(lambda y: preprocess2(y))
    
    return data2

def preprocess1(tweet):
    # this tweet tokenizer is used to separate each words and ponctuation in a sentence
    output = [x.strip().lower() for x in nltk.word_tokenize(tweet)]
    
    return " ".join(output)

def preprocess2(tweet):
    
    tknzr = TweetTokenizer(strip_handles=True)
    words = [x.strip().lower() for x in tknzr.tokenize(tweet)]

    # erase all the words that contains a ponctuation or other special signs but keep the one with an #
    output = [word for word in words if (word.isalpha() or word.startswith("#"))]
    
    # erase all the words contained in the nltk_words = the stopwords defined earlier
    # output = [w for w in words if not w in nltk_words]
    
    return " ".join(output)

def full_preprocessing(path_pos = "twitter-datasets/train_pos.txt", path_neg = "twitter-datasets/train_neg.txt",
                       path_test = "twitter-datasets/test_data.txt"):
    '''Makes the complete preprocessing on the dataset.
    Input:
     - path_pos : the path to the train_pos file
     - path_neg : the path to the train_neg file
     - path_test : the path to the test_data file

    Returns:
     - X_train : A list which contains the preprocessed training tweets
     - Y_train : A numpy.ndarray which contains the corresponding answer to each tweet (1 or 0)
                 More precisely, if Y_train[i] = 1, it means that the tweet X_train[i] used to contain a positive smiley
                                 if Y_train[i] = 0, it means that the tweet X_train[i] used to contain a negative smiley
     - Y_train_oh : A numpy.ndarray which maps Y_train (1 x nb_tweet) into a (2 x nb_tweet) matrix.
                    More precisely, if Y_train[i] = 1, then Y_train_oh[i] = [0. 1.]
                                    if Y_train[i] = 0, then Y_train_oh[i] = [1. 0.]
     - ids : 
     - X_test
     
     More precisely, the function full_preprocessing() operates the following steps :
     1) Remplace every n't into not to keep this information. Without it, the words with n't would just be erased.
        (Example : "I want but I can't" --> "I want but I cannot")
     2) Separate each word and punctuation by a space 
        (Example : "I'm soooExcited.!      to GO" --> "i ' m soooexcited . ! to go")
     3) Put every "# word" back to "#word"
        (Example : "So true # GoGo" --> "So true #Gogo")
     4) Erase each word containing a ponctuation or a special character EXCEPT the hastag symbol #
        (Example : "we 'clearly ha*) to do tah<3 #doit" --> "we to do #doit" )
     5) Fuse the positive and negative tweets and shuffle them
    '''
    
    # A) Open the data files and store them
    train_pos = read_data(path_pos)
    train_neg = read_data(path_neg)
    test = read_data(path_test)

    # B) Convert the variables into a panda.DataFrame
    train_pos_pd = pd.DataFrame(train_pos, columns=["tweet"])
    train_neg_pd = pd.DataFrame(train_neg, columns=["tweet"])
    test_pd = pd.DataFrame(test, columns=["tweet"])

    # C) Apply the complete preprocessing
    train_pos_preprocessed = pre_process_tweets(train_pos_pd)
    train_neg_preprocessed = pre_process_tweets(train_neg_pd)
    test_preprocessed = pre_process_tweets(test_pd)

    # D) Convert back the variables into list
    X_train_pos = list(train_pos_preprocessed.tweet)
    X_train_neg = list(train_neg_preprocessed.tweet)
    X_test = list(test_preprocessed.tweet)
    
    # E) Store positive and negative tweets in one variable (X_train) and create their corresponding answer (Y_train)
    X_train = X_train_pos + X_train_neg

    Y_train_pos = np.ones(len(X_train_pos), dtype = int)
    Y_train_neg = np.zeros(len(X_train_neg), dtype = int)
    Y_train = np.concatenate((Y_train_pos, Y_train_neg), axis = -1)

    # F) Shuffle the training tweets
    X_train, Y_train = shuffle(X_train, Y_train, random_state=52) #shuffle our training set
    
    # G) Compute Y_train_oh and ids
    Y_train_oh = convert_to_one_hot(Y_train, C=2)
    ids, _ = get_test_data(path_test)

    return X_train, Y_train, Y_train_oh, ids, X_test