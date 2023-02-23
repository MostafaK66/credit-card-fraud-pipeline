import numpy as np
import pandas as pd
from dtype_diet import optimize_dtypes, report_on_dataframe
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import RobustScaler

import settings


def read_data_as_data_frame(path_to_read_data):
    df_big = pd.read_csv(path_to_read_data)
    proposed_df = report_on_dataframe(df_big, unit="MB")
    df = optimize_dtypes(df_big, proposed_df)

    print(
        "No Fraud",
        round(df["Class"].value_counts()[0] / len(df) * 100, 2),
        "% of dataset",
    )

    print(
        "Fraud", round(df["Class"].value_counts()[1] / len(df) * 100, 2), "% of dataset"
    )

    return df


def make_robust_scaler(df):
    rob_scaler = RobustScaler()
    df["scaled_amount"] = rob_scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
    df["scaled_time"] = rob_scaler.fit_transform(df["Time"].values.reshape(-1, 1))

    return df


def drop_unnecessary_columns(df):
    df.drop(["Time", "Amount"], axis=1, inplace=True)
    scaled_amount = df["scaled_amount"]
    scaled_time = df["scaled_time"]
    df.drop(["scaled_amount", "scaled_time"], axis=1, inplace=True)
    df.insert(0, "scaled_amount", scaled_amount)
    df.insert(1, "scaled_time", scaled_time)

    return df


def make_stratified_split(df, stratified_splits):
    stratified_spliter = StratifiedKFold(
        n_splits=stratified_splits, random_state=None, shuffle=False
    )
    X = df.drop("Class", axis=1)
    y = df["Class"]

    for train_index, test_index in stratified_spliter.split(X, y):
        print("Train:", train_index, "Test: ", test_index)
        original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
        original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

    return (
        original_Xtrain,
        original_Xtest,
        original_ytrain,
        original_ytest,
        stratified_spliter,
    )


def check_target_distribution(original_ytrain, original_ytest):
    train_unique_label, train_counts_label = np.unique(
        original_ytrain, return_counts=True
    )
    test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

    print("Label Distribution: \n")
    print(train_counts_label / len(original_ytrain))
    print(test_counts_label / len(original_ytest))


def make_sub_sample_data_frame(df):
    fraud_df = df.loc[df["Class"] == 1]
    non_fraud_df = df.loc[df["Class"] == 0][:492]

    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
    df_new = normal_distributed_df.sample(frac=1, random_state=123)

    return normal_distributed_df, df_new


def check_target_distribution_sample_data_frame(df_new):
    print("Distribution of the classes in the subsample dataset:")
    print(df_new["Class"].value_counts() / len(df_new))


def make_outlier_removal(df_new, outlier_thre, column_name):
    V_fraud = df_new[column_name].loc[df_new["Class"] == 1].values
    q25, q75 = np.percentile(V_fraud, 25), np.percentile(V_fraud, 75)
    V_iqr = q75 - q25

    V_cut_off = V_iqr * outlier_thre
    V_lower, V_upper = q25 - V_cut_off, q75 + V_cut_off
    outliers_to_remove = [x for x in V_fraud if x < V_lower or x > V_upper]

    df_new = df_new.drop(
        df_new[(df_new[column_name] > V_upper) | (df_new[column_name] < V_lower)].index
    )

    return df_new, outliers_to_remove


def make_train_and_test_split(df_new, train_test_split_ration):
    X = df_new.drop("Class", axis=1)
    y = df_new["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_test_split_ration, random_state=123
    )
    X_train, X_test, y_train, y_test = (
        X_train.values,
        X_test.values,
        y_train.values,
        y_test.values,
    )

    return X_train, X_test, y_train, y_test


def make_train_test_split_main_df(df, stratified_splits):
    under_sample_X = df.drop("Class", axis=1)
    under_sample_y = df["Class"]
    stratified_spliter = StratifiedKFold(
        n_splits=stratified_splits, random_state=None, shuffle=False
    )

    for train_index, test_index in stratified_spliter.split(
        under_sample_X, under_sample_y
    ):
        print("Train:", train_index, "Test:", test_index)
        under_sample_train_X, under_sample_test_X = (
            under_sample_X.iloc[train_index].values,
            under_sample_X.iloc[test_index].values,
        )
        under_sample_train_y, under_sample_test_y = (
            under_sample_y.iloc[train_index].values,
            under_sample_y.iloc[test_index].values,
        )

        return (
            under_sample_train_X,
            under_sample_test_X,
            under_sample_train_y,
            under_sample_test_y,
        )


def make_smote_sampling(
    original_Xtrain, original_ytrain, stratified_spliter, log_reg_params, num_cross_val
):

    rand_log_reg = RandomizedSearchCV(
        estimator=LogisticRegression(),
        param_distributions=log_reg_params,
        n_iter=num_cross_val,
    )

    original_Xtrain = original_Xtrain.values
    original_ytrain = original_ytrain.values
    for train_index, test_index in stratified_spliter.split(
        original_Xtrain, original_ytrain
    ):
        pipline_smote = imbalanced_make_pipeline(SMOTE(), rand_log_reg)
        smote_log_reg = pipline_smote.fit(
            X=original_Xtrain[train_index], y=original_ytrain[train_index]
        )
        best_model = rand_log_reg.best_estimator_

        return pipline_smote, smote_log_reg, best_model


def compute_scores_for_smote(
    best_model, pipline_smote, original_Xtrain, original_ytrain, stratified_spliter
):

    acc_smote = list()
    precision_smote = list()
    recall_smote = list()
    f1_smote = list()
    auc_smote = list()
    original_Xtrain = original_Xtrain.values
    original_ytrain = original_ytrain.values
    for train_index, test_index in stratified_spliter.split(
        original_Xtrain, original_ytrain
    ):
        prediction_smote = best_model.predict(original_Xtrain[test_index])
        acc_smote.append(
            pipline_smote.score(
                original_Xtrain[test_index], original_ytrain[test_index]
            )
        )
        precision_smote.append(
            precision_score(original_ytrain[test_index], prediction_smote)
        )
        recall_smote.append(recall_score(original_ytrain[test_index], prediction_smote))
        f1_smote.append(f1_score(original_ytrain[test_index], prediction_smote))
        auc_smote.append(roc_auc_score(original_ytrain[test_index], prediction_smote))

        return acc_smote, precision_smote, recall_smote, f1_smote, auc_smote
