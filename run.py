import numpy as np
import pandas as pd

import run_fct
import pre_processing
import LSTM_functions

import tensorflow.keras as keras
from keras.models import load_model

# Load the 200-d twitter dictionnary 
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs_only_alpha('dictionnary/glove.twitter.27B.200d.txt')

# Maximum length of the sentence (computed on the training set)
max_length = 11

# Load the testing set
ids = get_ids("twitter-datasets/test.txt")
X_test = read_data("twitter-datasets/test.txt")

# Pre-process the testing set
############

# Convert the testing set into its indices
X_test_indices = sentences_to_indices(X_test, word_to_index, max_length)

# Load the model architecture and weights (previously trained on the training set)
model = load_model('model.h5')

# Predict the label based on model 
label = predict_lstm(model, X_test_indices)

# Create submission file
create_csv_submission(ids, label, 'submission.csv')