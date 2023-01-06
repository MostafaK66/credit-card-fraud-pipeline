import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold


def read_data_as_data_frame(path_to_read_data):
    df = pd.read_csv(path_to_read_data)
    print("No Fraud", round(df["Class"].value_counts()[0]/len(df) * 100, 2), "% of dataset")
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

