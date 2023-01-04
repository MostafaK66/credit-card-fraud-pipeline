import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_amount_and_time_distribution(df, output_path):
    fig, ax = plt.subplots(1, 2, figsize=(18, 4))

    amount_val = df["Amount"].values
    time_val = df["Time"].values

    sns.histplot(amount_val, stat="density", kde=True, ax=ax[0], bins=50, color="r")
    ax[0].set_title("Density of Transaction Amount", fontsize=14)
    ax[0].set_xlabel("amount of transactions", fontsize=14)
    ax[0].set_xlim([min(amount_val), max(amount_val)])

    sns.histplot(time_val, ax=ax[1], stat="density", kde=True, color='b')
    ax[1].set_title('Distribution of Transaction Time', fontsize=14)
    ax[1].set_xlabel("amount of time", fontsize=14)
    ax[1].set_xlim([min(time_val), max(time_val)])

    plt.savefig(os.path.join(output_path, "amount_and_time_density.png"))

