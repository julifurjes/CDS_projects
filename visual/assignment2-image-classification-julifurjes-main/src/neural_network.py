import sys
sys.path.append(".")
from utils.preprocessing2 import loading_data, preprocessing # importing the preprocessing file
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

def classifying(X_train_dataset, X_test_dataset, y_train, y_test, labels):
    clf = MLPClassifier(random_state=42,
                    hidden_layer_sizes=(64, 10),
                    learning_rate="adaptive",
                    early_stopping=True,
                    verbose=True,
                    max_iter=20).fit(X_train_dataset, y_train)
    y_pred = clf.predict(X_test_dataset)
    report = classification_report(y_test, 
                               y_pred, 
                               target_names=labels)
    f = open("out/neural_classifier.txt", "w") 
    f.write(report) # saving the classifier metrics as a txt file
    f.close() # closing the txt file

def main():
    X_train, y_train, X_test, y_test = loading_data()
    X_train_dataset, X_test_dataset, labels = preprocessing(X_train, X_test)
    classifying(X_train_dataset, X_test_dataset, y_train, y_test, labels)

if __name__ == '__main__':
    main()