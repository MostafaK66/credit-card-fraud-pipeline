from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import time


def make_tsne_dim_reductor(df_new, n_components):
    X = df_new.drop("Class", axis=1)
    time_start = time.time()
    X_reduced_tsne = TSNE(n_components=n_components, random_state=123).fit_transform(X.values)
    time_end = time.time()
    print("T-SNE took {:.2}s".format(time_end - time_start))

    return X_reduced_tsne


def make_pca_dim_reductor(df_new, n_components):
    X = df_new.drop("Class", axis=1)
    time_start = time.time()
    X_reduced_pca = PCA(n_components=n_components, random_state=123).fit_transform(X.values)
    time_end = time.time()
    print("PCA took {:.2}s".format(time_end - time_start))

    return X_reduced_pca


def make_truncated_svd_dim_reductor(df_new, n_components, svd_algorithm):
    X = df_new.drop("Class", axis=1)
    time_start = time.time()
    X_reduced_truncated_svd = TruncatedSVD(
        n_components=n_components, algorithm=svd_algorithm, random_state=123
    ).fit_transform(X.values)
    time_end = time.time()
    print("Truncated_SGD took {:.2}s".format(time_end - time_start))

    return X_reduced_truncated_svd



