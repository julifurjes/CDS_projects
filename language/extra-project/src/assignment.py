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
    # first we fit to the training data...
    X_train_feats = vectorizer.fit_transform(X_train)
    #... then do it for our test data
    X_test_feats = vectorizer.transform(X_test)
    # get feature names
    feature_names = vectorizer.get_feature_names_out()
    return vectorizer, X_train_feats, X_test_feats

def classifying(X_train_feats, X_test_feats, y_train, y_test):
    classifier = MLPClassifier(activation = "logistic",
                           hidden_layer_sizes = (30,),
                           max_iter=1000,
                           random_state = 42) # the random state is neccessary for reproduction
    classifier.fit(X_train_feats, y_train)
    y_pred = classifier.predict(X_test_feats)
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    f = open("out/neural_network_classifier.txt", "w") # creating a txt file
    f.write(classifier_metrics) # saving the classifier metrics as a txt file
    f.close() # closing the txt file
    return classifier, y_pred

def emotion_class(data):
    classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)
    i = 0
    for post in data['cleaned_text']:
        prediction = classifier(post)
        # creating an empty list for each emotion
        anger = disgust = fear = joy = neutral = sadness = surprise = [0]
        anger = prediction[0].get("score")
        disgust = prediction[1].get("score")
        fear = prediction[2].get("score")
        joy = prediction[3].get("score")
        neutral = prediction[4].get("score")
        sadness = prediction[5].get("score")
        surprise = prediction[6].get("score")
        emotions = {anger:"anger",disgust:"disgust",fear:"fear",joy:"joy",neutral:"neutral",sadness:"sadness",surprise:"surprise"}
        data['top_emotion'].iloc[i] = emotions.get(max(emotions))
        i = i + 1
        print(data)
        return data

def emotion_class(data):
    classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)
    posts_list = data['cleaned_text'].astype(str).values.tolist()
    anger = disgust = fear = joy = neutral = sadness = surprise = [0] * len(posts_list)
    prediction = classifier(posts_list)
    # creating an empty list for each emotion
    for i in range(len(posts_list)): # saving the scores
        anger[i] = preds[i][0].get("score")
        disgust[i] = preds[i][1].get("score")
        fear[i] = preds[i][2].get("score")
        joy[i] = preds[i][3].get("score")
        neutral[i] = preds[i][4].get("score")
        sadness[i] = preds[i][5].get("score")
        surprise[i] = preds[i][6].get("score")
        emotions = {anger[i]:"anger",disgust[i]:"disgust",fear[i]:"fear",joy[i]:"joy",neutral[i]:"neutral",sadness[i]:"sadness",surprise[i]:"surprise"}
        data['top_emotion'].iloc[i] = emotions.get(max(emotions))    
    print(data)
    return data

def make_plot(data):
    sns.countplot(x=data['top_emotion']) # saving all titles
    plt.savefig('out/all_types.png') # saving the output

def main():
    data, X, y = prep_data()
    X_train, X_test, y_train, y_test = splitting_data(X, y)
    vectorizer, X_train_feats, X_test_feats = vectorizing(X_train, X_test)
    classifier, y_pred = classifying(X_train_feats, X_test_feats, y_train, y_test)
    data = emotion_class(data)
    make_plot(data)

main()