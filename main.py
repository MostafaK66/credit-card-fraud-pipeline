import preprocessing
import settings
import plotting


def main():
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

    # print(normal_distributed_df.head())
    print(outliers_to_remove_V14)
    print(outliers_to_remove_V12)
    print(outliers_to_remove_V10)
    print(df_new.head())


if __name__ == '__main__':
    main()



