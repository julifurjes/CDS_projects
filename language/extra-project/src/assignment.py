import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from nrclex import NRCLex
from transformers import pipeline
import tensorflow
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

# predefining functions

def clean_string(text):
    final_string = ""
    text = text.lower() # lower casing
    text = re.sub(r'\n', '', text) # removing line breaks
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator) # removing punctuation
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english") # defining stopwords
    #useless_words = useless_words + ['hi', 'im']
    text_filtered = [word for word in text if not word in useless_words] # removing stopwords
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered] # removing numbers
    final_string = ' '.join(text_filtered)
    return final_string

# running functions

def prep_data():
    data = pd.read_csv("data/mbti_1.csv") # importing dataset
    print(data.head())
    data['cleaned_text'] = data['posts'].apply(clean_string) # cleaning the posts
    print(data.head())
    X = data['cleaned_text'] #saving the post column as X
    skencoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
    labels_to_vectorise = data.type.values
    y_converted = skencoder.fit_transform(labels_to_vectorise.reshape(-1,1)) # converting the labels into vectors
    y = y_converted #saving the converted type column as y
    return data, X, y

def splitting_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,           # texts for the model
                                                    y,          # classification labels
                                                    test_size=0.2,   # create an 80/20 split
                                                    random_state=42) # random state for reproducibility
    return X_train, X_test, y_train, y_test

def vectorizing(X_train, X_test):
    vectorizer = TfidfVectorizer(ngram_range = (1,2),     # unigrams and bigrams (1 word and 2 word units)
                             max_df = 0.95,           # remove very common words
                             min_df = 0.05,           # remove very rare words
                             max_features = 100)      # keep only top 100 features
    X_train_feats = vectorizer.fit_transform(X_train) # fitting to the training data
    X_test_feats = vectorizer.transform(X_test) # fitting to the test data
    feature_names = vectorizer.get_feature_names_out() # get feature names
    return vectorizer, X_train_feats, X_test_feats

def classifying(X_train_feats, X_test_feats, y_train, y_test):
    classifier = MLPClassifier(activation = "logistic",
                           hidden_layer_sizes = (30,),
                           max_iter=1000,
                           random_state = 42) # the random state is neccessary for reproduction
    classifier.fit(X_train_feats, y_train)
    y_pred = classifier.predict(X_test_feats)
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    f = open("out/neural_network_classifier.txt", "w") # creating a txt file for the report
    f.write(classifier_metrics) # saving the classifier metrics as a txt file
    f.close() # closing the txt file
    return classifier, y_pred

def saving(vectorizer, classifier):
    dump(classifier, "out/neural_classifier.joblib") # saving the classifier
    dump(vectorizer, "out/tfidf_vectorizer.joblib") # saving the vectorizer

def loading():
    loaded_clf = load("out/neural_classifier.joblib") # loading the classifier
    loaded_vect = load("out/tfidf_vectorizer.joblib") # loaading the vectorizer
    sentence = input("Enter a sentence: ") # asking for an input sentence
    test_sentence = loaded_vect.transform([sentence]) # vectorising
    pred = loaded_clf.predict(test_sentence) # predicting
    print("PREDICTION: ", np.argmax(pred))

def main():
    data, X, y = prep_data()
    X_train, X_test, y_train, y_test = splitting_data(X, y)
    vectorizer, X_train_feats, X_test_feats = vectorizing(X_train, X_test)
    classifier, y_pred = classifying(X_train_feats, X_test_feats, y_train, y_test)
    saving(vectorizer, classifier)
    loading()

if __name__ == '__main__':
    main()
