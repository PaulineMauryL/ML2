import numpy as np 


def get_data(neg, pos):
    with open(neg, "r") as file: #read only
        #tweets = list()
        tweets = str()

        for idx ,line in enumerate(file):
            #tweets.append(line)
            tweets+= line

    tweets = tweets.split('\n')
    del tweets[-1] #deletes last item

    tweets_neg = tweets

    with open(pos, "r") as file: #read only
        #tweets = list()
        tweets = str()

        for idx ,line in enumerate(file):
            #tweets.append(line)
            tweets+= line

    tweets = tweets.split('\n')
    del tweets[-1] #deletes last item

    tweets_pos = tweets

    tweets_pos.extend(tweets_neg)
    X = tweets_pos

    Y_one = [1 for _ in range(int(len(X)/2))]
    Y_minus_one = [-1 for _ in range(int(len(X)/2)) ]
    Y = [*Y_one, *Y_minus_one]
    
    return X, Y

def train_prepro(X,Y):
    """remove the tweets that are identical
       --> we do it only for the train (and not the validation)
    """
    for i in range(len(X)-1):
        while X[i] == X[i+1]:
            X.pop(i+1)
            Y.pop(i+1)
    
    Y.index(-1)
    one = np.ones(Y.index(-1), dtype = int)
    minus_one = (-1)*np.ones( len(Y) - Y.index(-1) , dtype = int)
    Y = np.r_[one, minus_one]
    
    return X,Y

def one_hot(Y):
    Y_hot = np.empty([len(Y), 2])
    Y_hot[Y== 1] = [1, 0]
    Y_hot[Y==-1] = [0, 1]
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
    
    # Average the word vectors
    for w in words:
        
        if w in word_to_vec_map.keys():
            avg += word_to_vec_map[w]
            
    avg = avg/len(words)
    
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
    #labels = np.zeros((m, 1))
    
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
            
        #if Y[j] == 1:
        #    labels[j] = 0
        #else:
        #    labels[j] = 1
        
    print("Accuracy: "  + str(np.mean((pred[:] == Y.reshape(Y.shape[0],1)[:])))) #== labels[:]))))
    
    return pred