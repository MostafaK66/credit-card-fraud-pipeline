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

    # print(normal_distributed_df.head())
    # print(df_new.head())


if __name__ == '__main__':
    main()



