import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from joblib import dump, load

def import_file():
    global X
    global y
    filename = os.path.join("in/fake_or_real_news.csv")
    data = pd.read_csv(filename, index_col=0)
    X = data["text"]
    y = data["label"]

import_file()
    
def train_test():
    global X_train
    global X_test
    global y_train
    global y_test
    X_train, X_test, y_train, y_test = train_test_split(X,           # texts for the model
                                                    y,          # classification labels
                                                    test_size=0.2,   # create an 80/20 split
                                                    random_state=42) # random state for reproducibility

train_test()

def vectorizing():
    global vectorizer
    global X_train_feats
    global X_test_feats
    vectorizer = TfidfVectorizer(ngram_range = (1,2),     # unigrams and bigrams (1 word and 2 word units)
                             lowercase =  True,       # why use lowercase?
                             max_df = 0.95,           # remove very common words
                             min_df = 0.05,           # remove very rare words
                             max_features = 100)      # keep only top 100 features
    
    # first we fit to the training data...
    X_train_feats = vectorizer.fit_transform(X_train)

    #... then do it for our test data
    X_test_feats = vectorizer.transform(X_test)

    # get feature names
    feature_names = vectorizer.get_feature_names_out()
    
vectorizing()

def classifying():
    global classifier
    classifier = MLPClassifier(activation = "logistic",
                           hidden_layer_sizes = (30,),
                           max_iter=1000,
                           random_state = 42)
    classifier.fit(X_train_feats, y_train)
    y_pred = classifier.predict(X_test_feats)
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    f = open("out/neural_network_classifier.txt", "w")
    f.write(classifier_metrics)
    f.close()
    
classifying()

def saving_models():
    dump(classifier, "models/NN_classifier.joblib")
    dump(vectorizer, "models/NN_tfidf_vectorizer.joblib")

saving_models()