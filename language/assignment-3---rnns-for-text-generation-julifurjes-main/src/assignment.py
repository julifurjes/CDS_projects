# data processing tools
import string, os 
import pandas as pd
import numpy as np
from joblib import dump
np.random.seed(42)

# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

tokenizer = Tokenizer() # loading the tokenizer function

# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

from random import sample

from predefined_functions import *

def load_data():
    data_dir = os.path.join("data") # define filepath
    all_comments = [] # create empty list
    for filename in os.listdir(data_dir):
        if 'Comments' in filename: # use only the comments files
            article_df = pd.read_csv(data_dir + "/" + filename) #reading csv file
            all_comments.extend(list(article_df["commentBody"].values)) # pushing it to the list
    all_comments = [h for h in all_comments if h != "Unknown"] # getting rid of 'unknown' entries 
    print(len(all_comments)) # counting how many datapoints we have
    all_comments = sample(all_comments, 3000) #randomly sampling
    return all_comments
    
def tokenizing(data):
    corpus = [clean_text(x) for x in data] # creating a clean corpus
    tokenizer.fit_on_texts(corpus) # updating internal vocabulary
    total_words = len(tokenizer.word_index) + 1 # counting the amount of words
    input_sequences = get_sequence_of_tokens(tokenizer, corpus) # converting data into sequence of tokens
    predictors, label, max_sequence_len = generate_padded_sequences(input_sequences, total_words) # padding our input sentences
    
    # saving tokenizing
    if not os.path.exists("in"):
        os.makedirs("in") # create dir if it doesnt exist yet
    np.savez("in/data.npz", 
            predictors = predictors, 
            label = label, 
            max_sequence_len = max_sequence_len, 
            total_words = total_words) # save data
    with open("in/tokenizer.json", 'w', encoding='utf-8') as f:
        f.write(tokenizer.to_json()) # exporting to json
    return predictors, label, max_sequence_len, total_words, tokenizer

def creating_model(max_length, nr_words, pred, lab):
    model = create_model(max_length, nr_words)
    model.summary()
    history = model.fit(pred, 
                    lab, 
                    epochs=5,
                    batch_size=500, 
                    verbose=1)
    print (generate_text("Hungary", 5, model, max_length))
    return model

def save_model(our_model, our_tokenizer, our_max_seq_len):
    tf.keras.models.save_model(our_model, "my_model", overwrite=True, save_format=None) # saving the model
    dump(our_tokenizer, os.path.join("my_model", "tokenizer.joblib")) # saving tokenizers
    with open(os.path.join("my_model", "max_seq_len.txt"), "w") as f:
        f.write(str(our_max_seq_len)) # saving max seq len

def main():
    all_comments = load_data()
    predictors, label, max_sequence_len, total_words, tokenizer = tokenizing(all_comments)
    model = creating_model(max_sequence_len, total_words, predictors, label)
    save_model(model, tokenizer, max_sequence_len)
   
if __name__ == '__main__':
    main()