import sys
sys.path.append(".")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import metrics
from joblib import dump, load
from utils.vectorizer_file import import_file, train_test, vectorizing # importing the preprocessing file

def classifying(X_train_feats, y_train, X_test_feats, y_test):
    classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train) # the random state is neccessary for reproduction
    y_pred = classifier.predict(X_test_feats)
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    f = open("out/log_regression_classifier.txt", "w") 
    f.write(classifier_metrics) # saving the classifier metrics as a txt file
    f.close() # closing the txt file
    return classifier

# saving the models for a faster reproduction
def saving_models(classifier, vectorizer):
    dump(classifier, "models/LR_classifier.joblib")
    dump(vectorizer, "models/LR_tfidf_vectorizer.joblib")

def main():
    X, y = import_file()
    X_train, X_test, y_train, y_test = train_test(X, y)
    vectorizer, X_train_feats, X_test_feats = vectorizing(X_train, X_test)
    classifier = classifying(X_train_feats, y_train, X_test_feats, y_test)
    saving_models(classifier, vectorizer)

if __name__ == '__main__':
    main()