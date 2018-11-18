import numpy as np 
from sklearn.utils import shuffle
import csv
import pickle
# --------------------------------------------------------------------
# ----------------------- Preprocessing ------------------------------
# --------------------------------------------------------------------

def get_data(pos = "./datasets/train_pos.txt", neg = "./datasets/train_neg.txt"):
    X_pos = read_data(pos)
    X_neg = read_data(neg)
    X_train = X_pos + X_neg
    
    Y_pos = np.ones(len(X_pos), dtype = int)
    Y_neg = np.zeros(len(X_neg), dtype = int)
    Y_train = np.concatenate((Y_pos, Y_neg), axis = -1)
    
    X_train_shuffled, Y_train_shuffled = shuffle(X_train, Y_train, random_state=52)
    
    return X_train_shuffled, Y_train_shuffled

def read_data(data):
    with open(data, "r") as file:
        tweets = str()
        for _,line in enumerate(file):
            tweets += line
        tweets = tweets.split('\n')
        del tweets[-1]
    return tweets

def train_prepro(X,Y):
    """remove the tweets that are identical
       --> we do it only for the train (and not the validation)
    """
#    for i in range(len(X)-1):
#        while X[i] == X[i+1]:
#            print(X[i])
#            np.delete(X,i+1)
#            np.delete(Y,i+1)
    
    Y_second = Y.astype(int)
    
    return X,Y_second

# --------------------------------------------------------------------
# ---------------------- Word embedding ------------------------------
# --------------------------------------------------------------------

def read_glove_vecs(glove_file):
    with open(glove_file, 'r',encoding='utf8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
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



def read_embeddings_vecs(embeddings, vocabulary):
    with open(vocabulary, 'rb') as voc:
        vocab = pickle.load(voc)
        
    words_embeddings = np.load('embeddings.npy')      # (nb_words, embedding_dimension)
        
    words = []                          # on veut les mots que une fois
    word_to_vec_map = {}  
    
    for word, idx in vocab.items():  
        words.append(word)                 #only possible because dict is ordered
        word_to_vec_map[word] = words_embeddings[idx, :]
        
    i = 1
    words_to_index = {}
    index_to_words = {}
    for w in sorted(words):
        words_to_index[w] = i
        index_to_words[i] = w
        i = i + 1
    
    return words_to_index, index_to_words, word_to_vec_map

def sentence_to_avg(tweet, word_to_vec_map):
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
    avg = np.zeros(np.shape(list(word_to_vec_map.values())[0])[0],)
    
    nb = 0
    # Average the word vectors
    for w in words:
        if w in word_to_vec_map.keys():
            avg += word_to_vec_map[w]
            nb = nb + 1
    if nb > 0:
        avg = avg/nb
    
    return avg

# --------------------------------------------------------------------
# -------------------------- Modeling --------------------------------
# --------------------------------------------------------------------
def model(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 400):
    """
    Model to train word vector representations in numpy.
    
    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m, 1)
    Y -- labels, numpy array of integers between -1 and 1, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 20-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations
    
    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """
    
    np.random.seed(32)

    # Define number of training examples
    m = len(Y)                              # number of training examples
    n_y = 2                                 # number of classes  
    n_h = 20                                # dimensions of the embeddings vectors 
    
    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    
    # Convert Y to Y_onehot with n_y classes
    Y_oh = convert_to_one_hot(Y, n_y)
    
    # Optimization loop
    for t in range(num_iterations):                       # Loop over the number of iterations
        for i in range(m):                                # Loop over the training examples
        
            # Average the word vectors of the words from the i'th training example
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # Forward propagate the avg through the softmax layer
            z = W @ avg + b
            a = softmax(z)
            
            # Compute cost using the i'th training label's one hot representation and "A" (the output of the softmax)
            cost = - np.sum(Y_oh[i]*np.log(a))
            
            # Compute gradients 
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz

            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db
        
        if t % 10 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map)
    return W, b

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

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



def predict_test(X, W, b, word_to_vec_map):
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
            
    return pred


def create_csv_submission(ids, y_pred, name):
    """
    Function taken from helpers of project 1
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

            
def get_test_data(data):
    with open(data, "r") as file:
        X_test = []
        ids = []
        for _, line in enumerate(file):
            ids.append( line.split(',', 1)[0] )
            X_test.append(" ".join(line.split(',', 1)[1:] ))
    return ids, X_test
        