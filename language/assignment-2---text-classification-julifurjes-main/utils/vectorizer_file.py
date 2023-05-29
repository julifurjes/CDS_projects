import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, ShuffleSplit

# importing our dataset
def import_file():
    filename = os.path.join("data/fake_or_real_news.csv")
    data = pd.read_csv(filename, index_col=0)
    X = data["text"] #saving the news column as a variable
    y = data["label"] #saving the label column as a variable
    return X, y

def train_test(X_data, y_data):
    X_train, X_test, y_train, y_test = train_test_split(X_data,           # texts for the model
                                                    y_data,          # classification labels
                                                    test_size=0.2,   # create an 80/20 split
                                                    random_state=42) # random state for reproducibility
    return X_train, X_test, y_train, y_test

def vectorizing(X_train_data, X_test_data):
    vectorizer = TfidfVectorizer(ngram_range = (1,2),     # unigrams and bigrams (1 word and 2 word units)
                             lowercase =  True,       # why use lowercase?
                             max_df = 0.95,           # remove very common words
                             min_df = 0.05,           # remove very rare words
                             max_features = 100)      # keep only top 100 features
    # first we fit to the training data...
    X_train_feats = vectorizer.fit_transform(X_train_data)
    #... then do it for our test data
    X_test_feats = vectorizer.transform(X_test_data)
    # get feature names
    feature_names = vectorizer.get_feature_names_out()
    return vectorizer, X_train_feats, X_test_feats

def main():
    X, y = import_file()
    X_train, X_test, y_train, y_test = train_test(X, y)
    vectorizer, X_train_feats, X_test_feats = vectorizing(X_train, X_test)
    return X, y, X_train, X_test, y_train, y_test, vectorizer, X_train_feats, X_test_feats

if __name__ == '__main__':
    X, y, X_train, X_test, y_train, y_test, vectorizer, X_train_feats, X_test_feats = main()