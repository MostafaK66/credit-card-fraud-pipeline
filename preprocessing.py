import pandas as pd


def read_data_as_data_frame(path_to_read_data):
    df = pd.read_csv(path_to_read_data)
    print("No Fraud", round(df["Class"].value_counts()[0]/len(df) * 100, 2), "% of dataset")
    print("Fraud", round(df["Class"].value_counts()[1] / len(df) * 100, 2), "% of dataset")

    return df
