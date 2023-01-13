import matplotlib.pyplot as plt
import seaborn as sns
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
