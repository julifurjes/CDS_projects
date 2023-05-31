from sklearn.linear_model import LogisticRegression

def show_features(vectorizer, X_train_feats, training_labels, classifier, n=5):
    """
    Return the most informative features from a classifier, i.e. the 'strongest' predictors.
    
    vectorizer:
        A vectorizer defined by the user, e.g. 'CountVectorizer'
    classifier:
        A classifier defined by the user, e.g. 'MultinomialNB'
    n:
        Number of features to display, defaults to 20
        
    """
    # Get feature names and coefficients
    feature_names = vectorizer.get_feature_names_out()
    classifier.fit(X_train_feats, training_labels)
    coefs_with_fns = sorted(zip(classifier.coef_[0], feature_names))
    # Get ordered labels
    labels = sorted(set(training_labels))
    # Select top n results, where n is function argument
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    # Pretty print columns showing most informative features
    print(f"{labels[0]}\t\t\t\t{labels[1]}\n")
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print("%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))