import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
PATH_TO_READ_DATA = "/Users/mostafa_mac/Desktop/kaggle_datasets/creditcard.csv"
OUT_PUT_PATH = os.path.join(os.getcwd(), "output")
STRATIFIED_SPLITS = 5
COLORS_FOR_BOX_PLOT = ['#B3F9C5', '#f9c5b3']
OUTLIER_THRE = 1.5
NUM_COMPONENTS = 2
SVD_ALGORITHM = "randomized"
TRAIN_TEST_SPLIT_RATIO = 0.3
dict_classifiers = {
    "LogisticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "SupportVectorClassifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier()
}
NUM_CROSS_VAL = 5