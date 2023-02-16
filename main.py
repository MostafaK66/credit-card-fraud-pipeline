import time
import warnings

from sklearn.linear_model import LogisticRegression

import classifiers
import dimensionality_reductors as dim_reduct
import model_validation as mv
import plotting
import preprocessing
import settings

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
    log_classifier_params, log_classifier = classifiers.make_classifier_with_grid_search(
        X_train=X_train,
        y_train=y_train,
        classifier_name=LogisticRegression(),
        classifier_params=settings.log_reg_params
    )
    # knears_classifier_params, knears_classifier = classifiers.make_classifier_with_grid_search(
    #     X_train=X_train,
    #     y_train=y_train,
    #     classifier_name=KNeighborsClassifier(),
    #     classifier_params=settings.knears_params
    # )
    # svc_classifier_params, svc_classifier = classifiers.make_classifier_with_grid_search(
    #     X_train=X_train,
    #     y_train=y_train,
    #     classifier_name=SVC(),
    #     classifier_params=settings.svc_params
    # )
    # tree_classifier_params, tree_classifier = classifiers.make_classifier_with_grid_search(
    #     X_train=X_train,
    #     y_train=y_train,
    #     classifier_name=DecisionTreeClassifier(),
    #     classifier_params=settings.tree_params
    # )
    preprocessing.make_train_test_split_main_df(
        df=df,
        stratified_splits=settings.STRATIFIED_SPLITS
    )
    under_sample_train_X, under_sample_test_X, under_sample_train_y, under_sample_test_y = \
        preprocessing.make_train_test_split_main_df(
            df=df,
            stratified_splits=settings.STRATIFIED_SPLITS
        )
    train_size_lr, train_scores_lr, test_scores_lr = mv.make_learning_curve(
        model_cls=log_classifier,
        df_new=df_new,
        num_split_cv=settings.NUM_SPLIT_CV,
        train_test_split_ratio=settings.TRAIN_TEST_SPLIT_RATIO,
        train_sizes_learning_curve=settings.TRAIN_SIZES_LEARNING_CURVE
    )

    # train_size_kn, train_scores_kn, test_scores_kn = mv.make_learning_curve(
    #     model_cls=knears_classifier,
    #     df_new=df_new,
    #     num_split_cv=settings.NUM_SPLIT_CV,
    #     train_test_split_ratio=settings.TRAIN_TEST_SPLIT_RATIO,
    #     train_sizes_learning_curve=settings.TRAIN_SIZES_LEARNING_CURVE
    # )
    #
    # train_size_svc, train_scores_svc, test_scores_svc = mv.make_learning_curve(
    #     model_cls=svc_classifier,
    #     df_new=df_new,
    #     num_split_cv=settings.NUM_SPLIT_CV,
    #     train_test_split_ratio=settings.TRAIN_TEST_SPLIT_RATIO,
    #     train_sizes_learning_curve=settings.TRAIN_SIZES_LEARNING_CURVE
    # )
    # train_size_tree, train_scores_tree, test_scores_tree = mv.make_learning_curve(
    #     model_cls=tree_classifier,
    #     df_new=df_new,
    #     num_split_cv=settings.NUM_SPLIT_CV,
    #     train_test_split_ratio=settings.TRAIN_TEST_SPLIT_RATIO,
    #     train_sizes_learning_curve=settings.TRAIN_SIZES_LEARNING_CURVE
    # )

    lr_train_score_mean, lr_train_score_std, lr_test_score_mean, lr_test_score_std = \
        mv.calculate_mean_std_of_scores(
            model_train_score=train_scores_lr,
            model_test_score=test_scores_lr
        )
    # kn_train_score_mean, kn_train_score_std, kn_test_score_mean, kn_test_score_std = \
    #     mv.calculate_mean_std_of_scores(
    #         model_train_score=train_scores_kn,
    #         model_test_score=test_scores_kn
    #     )
    # svc_train_score_mean, svc_train_score_std, svc_test_score_mean, svc_test_score_std = \
    #     mv.calculate_mean_std_of_scores(
    #         model_train_score=train_scores_svc,
    #         model_test_score=test_scores_svc
    #     )
    # tree_train_score_mean, tree_train_score_std, tree_test_score_mean, tree_test_score_std = \
    #     mv.calculate_mean_std_of_scores(
    #         model_train_score=train_scores_tree,
    #         model_test_score=test_scores_tree
    #     )
    # plotting.plot_learning_curve(
    #     train_size_lr=train_size_lr, lr_train_score_mean=lr_train_score_mean, lr_train_score_std=lr_train_score_std,
    #     lr_test_score_mean=lr_test_score_mean, lr_test_score_std=lr_test_score_std,
    #     train_size_kn=train_size_kn, kn_train_score_mean=kn_train_score_mean, kn_train_score_std=kn_train_score_std,
    #     kn_test_score_mean=kn_test_score_mean, kn_test_score_std=kn_test_score_std,
    #     train_size_svc=train_size_svc, svc_train_score_mean=svc_train_score_mean,
    #     svc_train_score_std=svc_train_score_std, svc_test_score_mean=svc_test_score_mean,
    #     svc_test_score_std=svc_test_score_std,
    #     train_size_tree=train_size_tree, tree_train_score_mean=tree_train_score_mean,
    #     tree_train_score_std=tree_train_score_std, tree_test_score_mean=tree_test_score_mean,
    #     tree_test_score_std=tree_test_score_std,
    #     output_path=settings.OUT_PUT_PATH, ylim=None
    # )

    log_reg_pred = mv.calculate_cross_val_predict(
        model_name=log_classifier,
        X_train=under_sample_train_X,
        y_train=under_sample_train_y,
        cv=settings.NUM_CROSS_VAL,
        method=settings.CROSS_VAL_METHOD
    )
    # knears_reg_pred = mv.calculate_cross_val_predict(
    #     model_name=knears_classifier,
    #     X_train=under_sample_train_X,
    #     y_train=under_sample_train_y,
    #     cv=settings.NUM_CROSS_VAL,
    #     method="predict"
    # )
    # svc_reg_pred = mv.calculate_cross_val_predict(
    #     model_name=svc_classifier,
    #     X_train=under_sample_train_X,
    #     y_train=under_sample_train_y,
    #     cv=settings.NUM_CROSS_VAL,
    #     method=settings.CROSS_VAL_METHOD
    # )
    # tree_reg_pred = mv.calculate_cross_val_predict(
    #     model_name=tree_classifier,
    #     X_train=under_sample_train_X,
    #     y_train=under_sample_train_y,
    #     cv=settings.NUM_CROSS_VAL,
    #     method="predict"
    # )
    log_fbr, log_tpr, log_threshold = mv.calculate_roc_auc_score(
        model_prediction=log_reg_pred,
        y_train=under_sample_train_y,
        name_of_model="Logistic Regression Classifier"
    )
    # knn_fbr, knn_tpr, knn_threshold = mv.calculate_roc_auc_score(
    #     model_prediction=knears_reg_pred,
    #     y_train=under_sample_train_y,
    #     name_of_model="KNears Classifier"
    # )
    # svc_fbr, svc_tpr, svc_threshold = mv.calculate_roc_auc_score(
    #     model_prediction=svc_reg_pred,
    #     y_train=under_sample_train_y,
    #     name_of_model="Support Vector Classifier"
    # )
    # tree_fbr, tree_tpr, tree_threshold = mv.calculate_roc_auc_score(
    #     model_prediction=tree_reg_pred,
    #     y_train=under_sample_train_y,
    #     name_of_model="Decision Tree Classifier"
    # )
    # plotting.plot_roc_curve_all_models(
    #     log_fbr=log_fbr,
    #     log_tpr=log_tpr,
    #     knn_fbr=knn_fbr,
    #     knn_tpr=knn_tpr,
    #     svc_fbr=svc_fbr,
    #     svc_tpr=svc_tpr,
    #     tree_fbr=tree_fbr,
    #     tree_tpr=tree_tpr,
    #     output_path=settings.OUT_PUT_PATH
    # )
    log_precision, log_recall = mv.calculate_precision_recall_log_reg(
        y_test=y_test,
        X_test=X_test,
        log_classifier=log_classifier
    )
    print(log_precision)
    print(log_recall)

    time_end = time.time()
    print(f"Total running time: {time_end - time_start}")


if __name__ == '__main__':
    main()
