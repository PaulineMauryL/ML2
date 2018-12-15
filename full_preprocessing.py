#FOR PREPROCESSING : ==============================================
import numpy as np
import pandas as pd
import re
from our_functionsv3 import read_data
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
    '''Makes the complete preprocessing on the dataset
    Input:
     - path_pos : the path to the train_pos file
     - path_neg : the path to the train_neg file
     - path_test : the path to the test_data file

    Output: It creates the 3 preprocessed data files
     - train_pos_tokenize_not_hash.txt is the preprocessed training positive file
     - train_neg_tokenize_not_hash.txt is the preprocessed training negative file
     - test_preprocessed_tokenize_not_hash.txt is the preprocessed test_data file
     
     More precisely, the complete preprocessing operates the following steps :
     1) Remplace every n't into not to keep this information. Without it, the words with n't would just be erased.
        (Example : "I want but I can't" --> "I want but I cannot")
     2) Separate each word and punctuation by a space 
        (Example : "I'm soooExcited.!      to GO" --> "i ' m soooexcited . ! to go")
     3) Put every "# word" back to "#word"
        (Example : "So true # GoGo" --> "So true #Gogo")
     4) Erase each word containing a ponctuation or a special character EXCEPT the hastag symbol #
        (Example : "we 'clearly ha*) to do tah<3 #doit" --> "we to do #doit" )
     
    '''
    
    train_pos = read_data(path_pos)
    train_neg = read_data(path_neg)

    pos = pd.DataFrame(train_pos, columns=["tweet"])
    neg = pd.DataFrame(train_neg, columns=["tweet"])

    test = read_data(path_test)
    test_pd = pd.DataFrame(test, columns=["tweet"])

    train_pos_preprocessed = pre_process_tweets(pos)
    train_neg_preprocessed = pre_process_tweets(neg)

    train_pos_preprocessed.to_csv('twitter-datasets/train_pos_tokenize_not_hash.txt', header=None, index=False, sep='\t')
    train_neg_preprocessed.to_csv('twitter-datasets/train_neg_tokenize_not_hash.txt', header=None, index=False, sep='\t')

    test_preprocessed = pre_process_tweets(test_pd)
    test_preprocessed.to_csv('twitter-datasets/test_preprocessed_tokenize_not_hash.txt', header=None, index=False, sep='\t')