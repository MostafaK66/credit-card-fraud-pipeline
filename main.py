import preprocessing
import settings


def main():
    df = preprocessing.read_data_as_data_frame(
        path_to_read_data=settings.PATH_TO_READ_DATA
    )

    # print(df.head())


if __name__ == '__main__':
    main()



