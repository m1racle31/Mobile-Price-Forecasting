import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from load_data import read_csv_data  # Import function from load_data.py
from data_preprocessing import preprocess_data  # Import the preprocessing function


def calculate_correlation_matrix(df):
    # Drop irrelevant columns
    dropped = [
        "Mobile Depth (cm)",
        "Weight",
        "Screen Height",
        "Screen Weight",
        "Talk Time",
        "3G",
        "Touch Screen",
        "Wi-Fi",
    ]
    df.drop(dropped, axis=1, inplace=True)

    # Select all columns
    all_columns = list(df.columns)

    # Calculate Cramér's V correlation coefficient for each pair of variables
    correlation_matrix = pd.DataFrame(index=all_columns, columns=all_columns)

    for i in range(len(all_columns)):
        for j in range(len(all_columns)):
            if i == j:
                correlation_matrix.iloc[i, j] = (
                    1.0  # The correlation between a variable and itself is 1
                )
            else:
                # Create a contingency table for pairs of categorical variables
                contingency_table = pd.crosstab(df.iloc[:, i], df.iloc[:, j])
                # Calculate Cramér's V correlation coefficient
                chi2, _, _, _ = chi2_contingency(contingency_table)
                n = min(contingency_table.shape)
                correlation = (chi2 / (len(df) * (n - 1))) ** 0.5
                correlation_matrix.iloc[i, j] = correlation

    return correlation_matrix


def plot_correlation_matrix(correlation_matrix):
    # Plot a heatmap of the correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix.astype(float),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title("Correlation Matrix", fontweight="bold")
    plt.show()


if __name__ == "__main__":
    # Load data using the function from Load_data.py
    file_path = "../data/Mobile Price Classification.csv"
    df = read_csv_data(file_path)

    # Preprocess the data
    df = preprocess_data(df)

    # Calculate and plot the correlation matrix
    correlation_matrix = calculate_correlation_matrix(df)
    plot_correlation_matrix(correlation_matrix)
