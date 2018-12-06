# EPFL Machine Learning - Project 2 <br /> Option B : Text Sentiment Classification

The purpose of this project is to predict if a tweet message used to contain a positive :) or negative :( smiley, by considering only the remaining text.


## Prerequisites

1) Download [training and testing set](https://www.crowdai.org/challenges/epfl-ml-text-classification/dataset_files) from kaggle

2) Download tensorflow by either : <br />
   a) Create a new [anaconda environment with tensorflow](https://towardsdatascience.com/setup-an-environment-for-machine-learning-and-deep-learning-with-anaconda-in-windows-5d7134a3db10) following step 6. <br />
       - Open Anaconda Prompt <br />
       - Type the following command : <br />
             conda create -n tensorflow pip python=3.6 <br />
       - Then type the following command : <br />
             activate tensorflow <br />
   or b) Download [tensorflow](https://www.tensorflow.org/install/)

3) Install nltk :
    - Open Anaconda Prompt
    - Type the following command : <br />
          pip install nltk
4) Download [nltk](https://www.nltk.org/data.html)

5) Install stopwords :
    - Open Anaconda Prompt
    - Type the following command : <br />
          pip install stopwords
    
6) Install pandas : 
    - Open Anaconda Prompt
    - Type the following command : <br />
          pip install pandas
      
7) Install matplotlib : //PAS NECESSAIRE JE CROIS
    - Open Anaconda Prompt
    - Type the following command : <br />
          pip install matplotlib
            
8) Download [glove.twitter.27B.zip](https://nlp.stanford.edu/projects/glove/)

## Project files

### our_functionsv3.py
It contains necessary functions for
 - preprocessing the datas and creation of the submission.
 - words embedding
 - modeling

### LSTM_functions.py
It contains necessary functions to compute LSTM.

### TextPreprocessing(NLTK).ipynb

#### Prerequisites :
The files 'train_neg_full.txt','train_pos_full.txt' and 'test_data.txt' must be in a folder called 'twitter-dataset'.

#### Input :  the training and testing datas :
    - train_neg_full.txt
    - train_pos_full.txt
    - test_data.txt
#### Output :  the preprocessed training datas and the preprocessed testing datas :
    - train_neg_full_preprocessed.txt
    - train_pos_full_preprocessed.txt
    - test_preprocessed.txt

### LSTM-Clean.ipynb

#### Prerequisites :
The dictionnary 'glove.twitter.27B.200d.txt' must be in a folder called 'dictionnary'.

#### Input : the preprocessed training and testing datas :
    - train_neg_full_preprocessed.txt
    - train_pos_full_preprocessed.txt
    - test_preprocessed.txt
#### Output : The submission file :
    - submission_LSTM.csv
    

## Authors

* Audrey Jordan -*MT_MA3*-

* Pauline Maury Laribière -*MT_MA3*- [PaulineMauryL](https://github.com/PaulineMauryL/ML2)

* Jérôme Savary -*MT_MA1*- 


## Acknowledgments

