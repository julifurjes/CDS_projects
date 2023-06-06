import tensorflow as tf
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
tokenizer = Tokenizer() # loading the tokenizer function

from predefined_functions import *

def loading():
    loaded_model = tf.keras.models.load_model("my_model") # loading the model
    print(loaded_model.summary()) # looking at its summary
    filepath = os.path.join("my_model", "max_seq_len.txt")
    with open(filepath) as f:
        max_sequence_len = f.readlines() # loading max seq len
    f.close()
    max_sequence_len = int("".join(max_sequence_len)) # converting the list of string to string
    max_sequence_len = int(max_sequence_len) # converting string to int
    return loaded_model, max_sequence_len

def generating_text(model, seq_len):
    input_word = input('Give me a word: ') # asking for a word from participant
    model = tf.keras.models.load_model("my_model") # load model
    print ("Output: ", generate_text(input_word, 10, model, seq_len)) # printing the generated words

def main():
    loaded_model, max_sequence_len = loading()
    generating_text(loaded_model, max_sequence_len)

if __name__ == '__main__':
    main()