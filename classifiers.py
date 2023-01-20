from sklearn.model_selection import cross_val_score


def make_base_fraud_detector_classifiers(dict_classifiers, X_train, y_train, num_cross_val):
    for key, classifier in dict_classifiers.items():
        classifier.fit(X_train, y_train)
        training_score = cross_val_score(classifier, X_train, y_train, cv=num_cross_val)
        print(
            "Classifier:", classifier.__class__.__name__,
            "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score"
        )
