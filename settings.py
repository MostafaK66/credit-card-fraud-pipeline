import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# TODO:Revert changes on parameters
PATH_TO_READ_DATA = "/Users/mostafa_mac/Desktop/kaggle_datasets/creditcard.csv"
OUT_PUT_PATH = os.path.join(os.getcwd(), "output")
STRATIFIED_SPLITS = 5
COLORS_FOR_BOX_PLOT = ["#B3F9C5", "#f9c5b3"]
OUTLIER_THRE = 1.5
NUM_COMPONENTS = 2
SVD_ALGORITHM = "randomized"
TRAIN_TEST_SPLIT_RATIO = 0.3
DICT_CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "SupportVectorClassifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
}
NUM_CROSS_VAL = 5
LOG_REG_PARAMS = {"penalty": ["l1", "l2"], "C": [0.001]}
# LOG_REG_PARAMS = {"penalty": ["l1", "l2"], "C": [0.001]}
KNEARS_PARAMS = {
    "n_neighbors": list(range(2, 5, 1)),
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
}
# KNEARS_PARAMS = {"n_neighbors": list(range(2, 5, 1)), "algorithm": ["auto"]}
SVC_PARAMS = {"C": [0.5], "kernel": ["rbf", "poly"]}
# SVC_PARAMS = {"C": [0.5], "kernel": ["rbf"]}
TREE_PARAMS = {
    "criterion": ["gini", "entropy"],
    "max_depth": list(range(2, 4, 1)),
    "min_samples_leaf": list(range(5, 7, 1)),
}
NUM_SPLIT_CV = 100
TRAIN_SIZES_LEARNING_CURVE = np.linspace(0.1, 1.0, 5)
print(TRAIN_SIZES_LEARNING_CURVE)
CROSS_VAL_METHOD = "decision_function"
