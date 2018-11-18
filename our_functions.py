import numpy as np 
from sklearn.utils import shuffle

def read_data(data):
    with open(data, "r") as file:
        tweets = str()
        for _,line in enumerate(file):
            tweets += line
        tweets = tweets.split('\n')
        del tweets[-1]
    return tweets


def get_data_v2(pos = "./datasets/train_pos.txt", neg = "./datasets/train_neg.txt"):
    X_pos = read_data(pos)
    X_neg = read_data(neg)
    X_train = np.concatenate((X_pos, X_neg), axis = 0)
    
    Y_pos = np.ones(len(X_pos))
    Y_neg = np.zeros(len(X_neg))
    Y_train = np.concatenate((Y_pos, Y_neg), axis = 0)
    
    X_train_shuffled, Y_train_shuffled = shuffle(X_train, Y_train, random_state=0)
    
    return X_train, Y_train


def one_hot(Y):
    Y_hot = np.empty([len(Y), 2])
    Y_hot[Y== 1] = [1, 0]
    Y_hot[Y== 0] = [0, 1]
    return Y_hot

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



def read_embeddings_vecs(embeddings, vocabulary):
    with open(vocabulary, 'rb') as voc:
        vocab = pickle.load(voc)
    
    words_embeddings = np.load('embeddings.npy')      # (nb_words, embedding_dimension)
        
    words = set()                     # on veut les mots que une fois
    word_to_vec_map = {}  
    
    for word, idx in vocab.items():  
        words.add(word)                 #only possible because dict is ordered
        word_to_vec_map[word] = words_embeddings[idx, :]
        
    i = 1
    words_to_index = {}
    index_to_words = {}
    for w in sorted(words):
        words_to_index[w] = i
        index_to_words[i] = w
        i = i + 1
    
    return words_to_index, index_to_words, word_to_vec_map


def sentence_to_avg(tweet, word_to_vec_map,size=20):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.
    
    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 20-dimensional vector representation
    
    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (20,)
    """
    # Split sentence into list of lower case words
    words = [x.lower() for x in tweet.split()]
    # Initialize the average word vector
    avg = np.zeros((size,))

    nb = 0
    # Average the word vectors
    for w in words:
        if w in word_to_vec_map.keys():
            avg += word_to_vec_map[w]
            nb = nb + 1
    
    if nb > 0:
        avg = avg/nb
    
    return avg


def predict(X, Y, W, b, word_to_vec_map):
    """
    Given X (sentences) and Y (emoji indices), predict emojis and compute the accuracy of your model over the given set.
    
    Arguments:
    X -- input data containing sentences, numpy array of shape (m, None)
    Y -- labels, containing index of the label emoji, numpy array of shape (m, 1)
    
    Returns:
    pred -- numpy array of shape (m, 1) with your predictions
    """
    m = len(X)
    pred = np.zeros((m, 1))
    labels = np.zeros((m, 1))
    
    for j in range(m):                       # Loop over training examples
        
        # Split jth test example (sentence) into list of lower case words
        words = X[j].lower().split()
        
        # Average words' vectors
        avg = np.zeros((20,))
        
        nb = 0
        # Average the word vectors
        for w in words:
            if w in word_to_vec_map.keys():
                avg += word_to_vec_map[w]
                nb = nb + 1
               
        if nb > 0:
            avg = avg/nb

            # Forward propagation
            Z = np.dot(W, avg) + b
            A = softmax(Z)
            pred[j] = np.argmax(A)
        
    print("Accuracy: "  + str(np.mean((pred[:] == labels[:]))))#Y.reshape(Y.shape[0],1)[:]))))
    
    return pred