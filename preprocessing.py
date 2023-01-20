import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import train_test_split


def read_data_as_data_frame(path_to_read_data):
    df = pd.read_csv(path_to_read_data)
    print("No Fraud", round(df["Class"].value_counts()[0] / len(df) * 100, 2), "% of dataset")
    print("Fraud", round(df["Class"].value_counts()[1] / len(df) * 100, 2), "% of dataset")

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
    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, "scaled_amount", scaled_amount)
    df.insert(1, "scaled_time", scaled_time)

    return df


def make_stratified_split(df, stratified_splits):
    stratified_spliter = StratifiedKFold(n_splits=stratified_splits, random_state=None, shuffle=False)
    X = df.drop("Class", axis=1)
    y = df["Class"]

    for train_index, test_index in stratified_spliter.split(X, y):
        print("Train:", train_index, "Test: ", test_index)
        original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
        original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

    return original_Xtrain, original_Xtest, original_ytrain, original_ytest


def check_target_distribution(original_ytrain, original_ytest):
    train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
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

    df_new = df_new.drop(df_new[(df_new[column_name] > V_upper) | (df_new[column_name] < V_lower)].index)

    return df_new, outliers_to_remove


def make_train_and_test_split(df_new, train_test_split_ration):
    X = df_new.drop("Class", axis=1)
    y = df_new["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_ration, random_state=123)
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values

    return X_train, X_test, y_train, y_test
