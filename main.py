import preprocessing
import settings
import plotting
import dimensionality_reductors as dim_reduct
import classifiers
import warnings
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


def main():
    time_start = time.time()
    df = preprocessing.read_data_as_data_frame(
        path_to_read_data=settings.PATH_TO_READ_DATA
    )
    plotting.plot_amount_and_time_distribution(
        amount_val=df["Amount"].values,
        time_val=df["Time"].values,
        output_path=settings.OUT_PUT_PATH,
        plot_title="amount_and_time_density.png"
    )
    df = preprocessing.make_robust_scaler(
        df=df
    )
    plotting.plot_amount_and_time_distribution(
        amount_val=df["scaled_amount"].values,
        time_val=df["scaled_time"].values,
        output_path=settings.OUT_PUT_PATH,
        plot_title="scaled_amount_and_time_density.png"
    )

    df = preprocessing.drop_unnecessary_columns(
        df=df
    )
    original_Xtrain, original_Xtest, original_ytrain, original_ytest = preprocessing.make_stratified_split(
        df=df,
        stratified_splits=settings.STRATIFIED_SPLITS
    )
    preprocessing.check_target_distribution(
        original_ytrain=original_ytrain,
        original_ytest=original_ytest
    )
    normal_distributed_df, df_new = preprocessing.make_sub_sample_data_frame(
        df=df
    )
    preprocessing.check_target_distribution_sample_data_frame(
        df_new=df_new
    )
    plotting.plot_heat_map_for_data(
        df=df,
        df_new=df_new,
        output_path=settings.OUT_PUT_PATH
    )
    plotting.plot_box_plot_neg_corr(
        df_new=df_new,
        output_path=settings.OUT_PUT_PATH
    )

    plotting.plot_box_plot_pos_corr(
        df_new=df_new,
        output_path=settings.OUT_PUT_PATH
    )

    plotting.plot_distribution_neg_corr(
        df_new=df_new,
        output_path=settings.OUT_PUT_PATH
    )
    df_new, outliers_to_remove_V14 = preprocessing.make_outlier_removal(
        df_new=df_new,
        outlier_thre=settings.OUTLIER_THRE,
        column_name="V14"
    )
    df_new, outliers_to_remove_V12 = preprocessing.make_outlier_removal(
        df_new=df_new,
        outlier_thre=settings.OUTLIER_THRE,
        column_name="V12"
    )

    df_new, outliers_to_remove_V10 = preprocessing.make_outlier_removal(
        df_new=df_new,
        outlier_thre=settings.OUTLIER_THRE,
        column_name="V10"
    )
    plotting.plot_box_plot_reducted_outliers(
        df_new=df_new,
        output_path=settings.OUT_PUT_PATH,
        colors_box_plot=settings.COLORS_FOR_BOX_PLOT
    )

    X_reduced_tsne, y = dim_reduct.make_tsne_dim_reductor(
        df_new=df_new,
        n_components=settings.NUM_COMPONENTS
    )
    X_reduced_pca = dim_reduct.make_pca_dim_reductor(
        df_new=df_new,
        n_components=settings.NUM_COMPONENTS
    )

    X_reduced_truncated_svd = dim_reduct.make_truncated_svd_dim_reductor(
        df_new=df_new,
        n_components=settings.NUM_COMPONENTS,
        svd_algorithm=settings.SVD_ALGORITHM
    )
    plotting.plot_scatter_dim_reduction(
        X_reduced_tsne=X_reduced_tsne,
        X_reduced_pca=X_reduced_pca,
        X_reduced_truncated_svd=X_reduced_truncated_svd,
        y=y,
        output_path=settings.OUT_PUT_PATH
    )

    X_train, X_test, y_train, y_test = preprocessing.make_train_and_test_split(
        df_new=df_new,
        train_test_split_ration=settings.TRAIN_TEST_SPLIT_RATIO
    )
    classifiers.make_base_fraud_detector_classifiers(
        dict_classifiers=settings.dict_classifiers,
        X_train=X_train,
        y_train=y_train,
        num_cross_val=settings.NUM_CROSS_VAL
    )
    log_classifier = classifiers.make_classifier_with_grid_search(
        X_train=X_train,
        y_train=y_train,
        classifier_name=LogisticRegression(),
        classifier_params=settings.log_reg_params
    )
    knears_classifier = classifiers.make_classifier_with_grid_search(
        X_train=X_train,
        y_train=y_train,
        classifier_name=KNeighborsClassifier(),
        classifier_params=settings.knears_params
    )
    svc_classifier = classifiers.make_classifier_with_grid_search(
        X_train=X_train,
        y_train=y_train,
        classifier_name=SVC(),
        classifier_params=settings.svc_params
    )
    tree_classifier = classifiers.make_classifier_with_grid_search(
        X_train=X_train,
        y_train=y_train,
        classifier_name=DecisionTreeClassifier(),
        classifier_params=settings.tree_params
    )
    preprocessing.make_train_test_split_main_df(
        df=df,
        stratified_splits=settings.STRATIFIED_SPLITS
    )
    under_sample_train_X, under_sample_test_X, under_sample_train_y, under_sample_test_y =\
        preprocessing.make_train_test_split_main_df(
            df=df,
            stratified_splits=settings.STRATIFIED_SPLITS
        )
    print(len(under_sample_train_X))
    print(len(under_sample_test_X))
    print(len(under_sample_train_y))
    print(len(under_sample_test_y))
    time_end = time.time()
    print(f"Total running time: {time_end - time_start}")


if __name__ == '__main__':
    main()



