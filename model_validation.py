import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def make_learning_curve(model_cls, df_new, num_split_cv, train_test_split_ratio, train_sizes_learning_curve):
    #TODO: Remove slicing
    df_new = df_new.iloc[0:1000, :]
    X = df_new.drop("Class", axis=1)
    y = df_new["Class"]
    cv = ShuffleSplit(n_splits=num_split_cv, test_size=train_test_split_ratio, random_state=123)
    train_sizes, train_scores, test_scores = learning_curve(
        model_cls, X, y, cv=cv, train_sizes=train_sizes_learning_curve
    )

    return train_sizes, train_scores, test_scores


def calculate_mean_std_of_scores(model_train_score, model_test_score):
    model_train_score_mean = np.mean(model_train_score, axis=1)
    model_train_score_std = np.std(model_train_score, axis=1)
    model_test_score_mean = np.mean(model_test_score, axis=1)
    model_test_score_std = np.std(model_test_score, axis=1)

    return model_train_score_mean, model_train_score_std, model_test_score_mean, model_test_score_std



