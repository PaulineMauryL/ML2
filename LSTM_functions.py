import numpy as np
import pandas as pd
from plots_file import *
from our_functionsv3 import *

np.random.seed(0)
import tensorflow.keras as keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.callbacks import EarlyStopping
import time
import matplotlib
import matplotlib.pyplot as plt




def get_max_length(X_train):
    size = []
    for elem in X_train:
        size.append(len(elem.split()))
    max_length = max(size)
    return max_length



def sentences_to_indices(X, word_to_index, max_length):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """    
    m = len(X)                                   # number of training examples

    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_length))
    
    notfound=0
    
    for i in range(m):                               # loop over training examples        
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = [x.lower() for x in X[i].split()]
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            if w in word_to_index.keys():
            # Set the (i,j)th entry of X_indices to the index of the correct word.
                X_indices[i, j] = word_to_index[w]
                j = j + 1
            else:
                notfound = notfound + 1
                #X_indices[i, j] = -1   #si il ne connait pas le mot il met -1, voir comment on gère ça après
                # Increment j to j + 1
                #print("{}   not in twitter dataset".format(w))
            #j = j + 1
        '''    
        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
            j = j+1
        '''
    print("{} words were not in the dictionary".format(notfound))
    
    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    #print(vocab_len)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    #print(emb_matrix.shape)
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    #print(len(word_to_index  ))
    #print(len(word_to_vec_map))
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding(vocab_len, emb_dim, trainable = False) 

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer



#credits of function to http://parneetk.github.io/blog/neural-networks-in-keras/
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.savefig('graphs/history_early_stopping_21_11_not_preprocessed.png')
    plt.show()
    #savefig('history_early_stopping_20_11.png')


def smiley_LSTM(input_shape, word_to_vec_map, word_to_index, dropout_rate):
    """
    Function creating the Emojify-v2 model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape = input_shape, dtype = 'int32')
    
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences = True)(embeddings)
    
    # Add dropout with a probability of 0.5
    X = Dropout(dropout_rate)(X)
    
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128, return_sequences = False, return_state = False)(X)
    
    # Add dropout with a probability of 0.5
    X = Dropout(dropout_rate)(X)
    
    # Propagate X through a Dense layer with softmax activation to get back a batch of 2-dimensional vectors.
    X = Dense(2)(X)
    
    # Add a softmax activation
    X = Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs = sentence_indices, outputs = X)
    
    return model



def complete_model(X_train_indices, Y_train_oh, word_to_vec_map, word_to_index, max_length, summary = False, dropout_rate = 0.5, batch_size = 128, 
                   epochs = 50, loss ='categorical_crossentropy', optimizer ='adam'):
    
    model = smiley_LSTM((max_length,), word_to_vec_map, word_to_index, dropout_rate)
    
    if summary:
        model.summary()
        
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3, verbose=1, mode='auto')
    callbacks_list = [earlystop]
    
    start = time.time()
    history_lstm = model.fit(X_train_indices, Y_train_oh, epochs = 50, callbacks=callbacks_list, batch_size = 32, validation_split = 0.1, shuffle=True)
    end = time.time()
    print("Model took {} seconds (which is {} minutes or {} hours) to train".format((end - start), (end - start)/60, (end - start)/3600))
    
    return history_lstm, model


def predict_lstm(model, X_test_indices):
    y_pred = model.predict(X_test_indices)
    label = np.zeros([X_test_indices.shape[0], 1])
    
    for i in range(len(X_test_indices)):
        label[i] = np.argmax(y_pred[i])
    label[label == 0] = -1
    return label

def new_predict_lstm(model, X_test_indices):
    
    y_pred = model.predict(X_test_indices)
    label = np.zeros([X_test_indices.shape[0], 1])
    
    for i in range(len(X_test_indices)):
        if(y_pred[i] < 0.5):
            label[i] = -1
        else:
            label[i] = 1
    return label
