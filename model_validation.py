import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import ShuffleSplit, cross_val_predict, learning_curve


def make_learning_curve(
    model_cls, df_new, num_split_cv, train_test_split_ratio, train_sizes_learning_curve
):
    # TODO: Remove slicing
    df_new = df_new.iloc[0:1000, :]
    X = df_new.drop("Class", axis=1)
    y = df_new["Class"]
    cv = ShuffleSplit(
        n_splits=num_split_cv, test_size=train_test_split_ratio, random_state=123
    )
    train_sizes, train_scores, test_scores = learning_curve(
        model_cls, X, y, cv=cv, train_sizes=train_sizes_learning_curve
    )

    return train_sizes, train_scores, test_scores


def calculate_mean_std_of_scores(model_train_score, model_test_score):
    model_train_score_mean = np.mean(model_train_score, axis=1)
    model_train_score_std = np.std(model_train_score, axis=1)
    model_test_score_mean = np.mean(model_test_score, axis=1)
    model_test_score_std = np.std(model_test_score, axis=1)

    return (
        model_train_score_mean,
        model_train_score_std,
        model_test_score_mean,
        model_test_score_std,
    )


def calculate_cross_val_predict(model_name, X_train, y_train, cv, method):
    model_predicted_values = cross_val_predict(
        estimator=model_name, X=X_train, y=y_train, cv=cv, method=method
    )

    return model_predicted_values


def calculate_roc_auc_score(model_prediction, y_train, name_of_model):
    model_fbr, model_tpr, model_threshold = roc_curve(
        y_true=y_train, y_score=model_prediction
    )
    print(name_of_model, roc_auc_score(y_true=y_train, y_score=model_prediction))

    return model_fbr, model_tpr, model_threshold


def calculate_precision_recall_log_reg(y_test, X_test, log_classifier):
    under_sample_y_score = log_classifier.decision_function(X_test)
    log_precision, log_recall, _ = precision_recall_curve(y_test, under_sample_y_score)

    return log_precision, log_recall
