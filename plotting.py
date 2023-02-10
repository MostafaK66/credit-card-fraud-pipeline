import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import os


def plot_amount_and_time_distribution(amount_val, time_val, output_path, plot_title):
    fig, ax = plt.subplots(1, 2, figsize=(18, 4))

    sns.histplot(amount_val, stat="density", kde=True, ax=ax[0], bins=50, color="r")
    ax[0].set_title("Density of Transaction Amount", fontsize=14)
    ax[0].set_xlabel("amount of transactions", fontsize=14)
    ax[0].set_xlim([min(amount_val), max(amount_val)])

    sns.histplot(time_val, ax=ax[1], stat="density", kde=True, color='b')
    ax[1].set_title('Distribution of Transaction Time', fontsize=14)
    ax[1].set_xlabel("amount of time", fontsize=14)
    ax[1].set_xlim([min(time_val), max(time_val)])

    plt.savefig(os.path.join(output_path, plot_title))


def plot_heat_map_for_data(df, df_new, output_path):
    fig, ax = plt.subplots(1, 2, figsize=(18, 4))

    correlation_df = df.corr()
    sns.heatmap(correlation_df, ax=ax[0], cmap="coolwarm_r", annot_kws={'size': 20})
    ax[0].set_title("Imbalanced Correlation matrix \n (don't use for refrence)", fontsize=14)

    correlation_df_new = df_new.corr()
    sns.heatmap(correlation_df_new, ax=ax[1], cmap="coolwarm_r", annot_kws={'size': 20})
    ax[1].set_title("Subsample Correlation matrix \n (use for refrence)", fontsize=14)

    plt.savefig(os.path.join(output_path, "heatmap.png"))


def plot_box_plot_neg_corr(df_new, output_path):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(18, 4))

    sns.boxplot(x="Class", y="V17", data=df_new, ax=ax[0])
    ax[0].set_title("V17 vs Class - Negative Correlation")

    sns.boxplot(x="Class", y="V14", data=df_new, ax=ax[1])
    ax[1].set_title("V14 vs Class - Negative Correlation")

    sns.boxplot(x="Class", y="V12", data=df_new, ax=ax[2])
    ax[2].set_title("V12 vs Class - Negative Correlation")

    sns.boxplot(x="Class", y="V10", data=df_new, ax=ax[3])
    ax[3].set_title("V10 vs Class - Negative Correlation")

    plt.savefig(os.path.join(output_path, "boxplot_neg_corr.png"))


def plot_box_plot_pos_corr(df_new, output_path):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(18, 4))

    sns.boxplot(x="Class", y="V11", data=df_new, ax=ax[0])
    ax[0].set_title("V11 vs Class - Positive Correlation")

    sns.boxplot(x="Class", y="V4", data=df_new, ax=ax[1])
    ax[1].set_title("V4 vs Class - Positive Correlation")

    sns.boxplot(x="Class", y="V2", data=df_new, ax=ax[2])
    ax[2].set_title("V2 vs Class - Positive Correlation")

    sns.boxplot(x="Class", y="V19", data=df_new, ax=ax[3])
    ax[3].set_title("V19 vs Class - Positive Correlation")

    plt.savefig(os.path.join(output_path, "boxplot_pos_corr.png"))


def plot_distribution_neg_corr(df_new, output_path):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

    v14_fraud_list = df_new["V14"].loc[df_new["Class"] == 1].values
    sns.histplot(v14_fraud_list, stat="count", ax=ax[0], color="#FB8861")
    ax[0].set_title("V14 Distribution \n (Fraud Transaction)", fontsize=14)

    V12_fraud_dist = df_new["V12"].loc[df_new["Class"] == 1].values
    sns.histplot(V12_fraud_dist, stat="count", ax=ax[1], color="#56F9BB")
    ax[1].set_title("V12 Distribution \n (Fraud Transaction)", fontsize=14)

    V10_fraud_dist = df_new["V10"].loc[df_new["Class"] == 1].values
    sns.histplot(V10_fraud_dist, stat="count", ax=ax[2], color="#C5B3F9")
    ax[2].set_title("V10 Distribution \n (Fraud Transaction)", fontsize=14)

    plt.savefig(os.path.join(output_path, "dist_neg_err.png"))


def plot_box_plot_reducted_outliers(df_new, output_path, colors_box_plot):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

    sns.boxplot(x="Class", y="V14", data=df_new, ax=ax[0], palette=colors_box_plot)
    ax[0].set_title("V14 Feature \n Reduction of outliers", fontsize=14)
    ax[0].annotate("Fewer extreme \n outliers", xy=(0.98, -17.5), xytext=(0, -12), arrowprops=dict(facecolor='black'))

    sns.boxplot(x="Class", y="V12", data=df_new, ax=ax[1], palette=colors_box_plot)
    ax[1].set_title("V12 Feature \n Reduction of outliers", fontsize=14)
    ax[1].annotate("Fewer extreme \n outliers", xy=(0.98, -17.3), xytext=(0, -12), arrowprops=dict(facecolor='black'))

    sns.boxplot(x="Class", y="V10", data=df_new, ax=ax[2], palette=colors_box_plot)
    ax[2].set_title("V10 Feature \n Reduction of outliers", fontsize=14)
    ax[2].annotate("Fewer extreme \n outliers", xy=(0.95, -16.5), xytext=(0, -12), arrowprops=dict(facecolor='black'))

    plt.savefig(os.path.join(output_path, "boxplot_reducted_outliers.png"))


def plot_scatter_dim_reduction(X_reduced_tsne, X_reduced_pca, X_reduced_truncated_svd, y, output_path):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    fig.suptitle("Clustering using Dimensionality Reduction", fontsize=14)

    blue_patch = mpatches.Patch(color="#0A0AFF", label="No Fraud")
    red_patch = mpatches.Patch(color="#AF0000", label="Fraud")

    ax[0].scatter(
        X_reduced_tsne[:, 0], X_reduced_tsne[:, 1], c=(y == 0), cmap="coolwarm", label="No Fraud", linewidth=2
    )
    ax[0].scatter(
        X_reduced_tsne[:, 0], X_reduced_tsne[:, 1], c=(y == 1), cmap="coolwarm", label="Fraud", linewidth=2
    )
    ax[0].set_title("t-SNE", fontsize=14)
    ax[0].grid(True)
    ax[0].legend(handles=[blue_patch, red_patch])

    ax[1].scatter(
        X_reduced_tsne[:, 0], X_reduced_pca[:, 1], c=(y == 0), cmap="coolwarm", label="No Fraud", linewidth=2
    )
    ax[1].scatter(
        X_reduced_tsne[:, 0], X_reduced_pca[:, 1], c=(y == 1), cmap="coolwarm", label="Fraud", linewidth=2
    )
    ax[1].set_title("PCA", fontsize=14)
    ax[1].grid(True)
    ax[1].legend(handles=[blue_patch, red_patch])

    ax[2].scatter(
        X_reduced_tsne[:, 0], X_reduced_truncated_svd[:, 1], c=(y == 0), cmap="coolwarm", label="No Fraud", linewidth=2
    )
    ax[2].scatter(
        X_reduced_tsne[:, 0], X_reduced_truncated_svd[:, 1], c=(y == 1), cmap="coolwarm", label="Fraud", linewidth=2
    )
    ax[2].set_title("Truncated SVD", fontsize=14)
    ax[2].grid(True)
    ax[2].legend(handles=[blue_patch, red_patch])

    plt.savefig(os.path.join(output_path, "scatter_dim_reduc.png"))


def plot_learning_curve(
        train_size_lr, lr_train_score_mean, lr_train_score_std, lr_test_score_mean, lr_test_score_std,
        train_size_kn, kn_train_score_mean, kn_train_score_std, kn_test_score_mean, kn_test_score_std,
        train_size_svc, svc_train_score_mean, svc_train_score_std, svc_test_score_mean, svc_test_score_std,
        train_size_tree, tree_train_score_mean, tree_train_score_std, tree_test_score_mean, tree_test_score_std,
        output_path, ylim=None


):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14), sharey=True)
    if ylim is not None:
        plt.ylim(*ylim)
    ax1.fill_between(
        train_size_lr, lr_train_score_mean - lr_train_score_std, lr_train_score_mean + lr_train_score_std, alpha=0.1,
        color="#ff9124"
    )
    ax1.fill_between(
        train_size_lr, lr_test_score_mean - lr_test_score_std, lr_test_score_mean + lr_test_score_std, alpha=0.1,
        color="#2492ff"
    )
    ax1.plot(train_size_lr, lr_train_score_mean, "o-", color="#ff9124", label="Training score")
    ax1.plot(train_size_lr, lr_test_score_mean, "o-", color="#ff9124", label="Cross-validation score")
    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
    ax1.set_xlabel('Training size (m)')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend(loc="best")

    ax2.fill_between(
        train_size_kn, kn_train_score_mean - kn_train_score_std, kn_train_score_mean + kn_train_score_std, alpha=0.1,
        color="#ff9124"
    )
    ax2.fill_between(
        train_size_kn, kn_test_score_mean - kn_test_score_std, kn_test_score_mean + kn_test_score_std, alpha=0.1,
        color="#2492ff"
    )
    ax2.plot(train_size_kn, kn_train_score_mean, "o-", color="#ff9124", label="Training score")
    ax2.plot(train_size_kn, kn_test_score_mean, "o-", color="#ff9124", label="Cross-validation score")
    ax2.set_title("Knears Neighbors Learning Curve", fontsize=14)
    ax2.set_xlabel('Training size (m)')
    ax2.set_ylabel('Score')
    ax2.grid(True)
    ax2.legend(loc="best")

    ax3.fill_between(
        train_size_svc, svc_train_score_mean - svc_train_score_std, svc_train_score_mean + svc_train_score_std, alpha=0.1,
        color="#ff9124"
    )
    ax3.fill_between(
        train_size_svc, svc_test_score_mean - svc_test_score_std, svc_test_score_mean + svc_test_score_std, alpha=0.1,
        color="#2492ff"
    )
    ax3.plot(train_size_kn, kn_train_score_mean, "o-", color="#ff9124", label="Training score")
    ax3.plot(train_size_kn, kn_test_score_mean, "o-", color="#ff9124", label="Cross-validation score")
    ax3.set_title("Support Vector Classifier \n Learning Curve", fontsize=14)
    ax3.set_xlabel('Training size (m)')
    ax3.set_ylabel('Score')
    ax3.grid(True)
    ax3.legend(loc="best")

    ax4.fill_between(
        train_size_tree, tree_train_score_mean - tree_train_score_std, tree_train_score_mean + tree_train_score_std,
        alpha=0.1,
        color="#ff9124"
    )
    ax4.fill_between(
        train_size_tree, tree_test_score_mean - tree_test_score_std, tree_test_score_mean + tree_test_score_std,
        alpha=0.1, color="#2492ff"
    )
    ax4.plot(train_size_kn, kn_train_score_mean, "o-", color="#ff9124", label="Training score")
    ax4.plot(train_size_kn, kn_test_score_mean, "o-", color="#ff9124", label="Cross-validation score")
    ax4.set_title("Decision Tree Classifier \n Learning Curve", fontsize=14)
    ax4.set_xlabel('Training size (m)')
    ax4.set_ylabel('Score')
    ax4.grid(True)
    ax4.legend(loc="best")

    plt.savefig(os.path.join(output_path, "learning_curve.png"))


