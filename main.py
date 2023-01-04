import preprocessing
import settings
import plotting


def main():
    df = preprocessing.read_data_as_data_frame(
        path_to_read_data=settings.PATH_TO_READ_DATA
    )
    plotting.plot_amount_and_time_distribution(
        df=df,
        output_path=settings.OUT_PUT_PATH
    )

    print(df["Amount"].values)


if __name__ == '__main__':
    main()



