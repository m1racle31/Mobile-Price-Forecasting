import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from load_data import read_csv_data  # Import function from load_data.py
from data_preprocessing import preprocess_data  # Import the preprocessing function


# Base Feature Distribution
def price_distribution(df):
    # Data to be plotted
    price_range_counts = df["Price Range"].value_counts().sort_index()

    # Create a pie chart of the distribution of Price Range classes
    plt.figure(figsize=(8, 5))
    plt.pie(
        price_range_counts,
        labels=["Low Cost", "Medium Cost", "High Cost", "Very High Cost"],
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.title("Distribution of Price Ranges")
    plt.show()


def int_memory_distribution(df):
    int_memory = df["Internal Memory (GB)"]

    # Creating a histogram
    plt.figure(figsize=(8, 5))
    plt.hist(int_memory, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Internal Memory (GB)")
    plt.ylabel("Frequency")
    plt.title("Internal Memory Distribution")

    # Displaying the plot
    plt.show()


def ram_distribution(df):
    # Data to be plotted
    ram = df["RAM (MB)"]

    # Creating a histogram
    plt.figure(figsize=(8, 5))
    plt.hist(ram, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("RAM (MB)")
    plt.ylabel("Frequency")
    plt.title("RAM Distribution")

    # Displaying the plot
    plt.show()


def battery_distribution(df):
    # Data to be plotted
    battery = df["Battery Power"]

    # Creating a histogram
    plt.figure(figsize=(8, 5))
    plt.hist(battery, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Battery Power")
    plt.ylabel("Frequency")
    plt.title("Battery Power Distribution")

    # Displaying the plot
    plt.show()


def clk_speed_distribution(df):
    # Data to be plotted
    clock = df["Clock Speed"]

    # Creating a histogram
    plt.figure(figsize=(8, 5))
    plt.hist(clock, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Clock Speed")
    plt.ylabel("Frequency")
    plt.title("Clock Speed Distribution")

    # Displaying the plot
    plt.show()


def cpu_distribution(df):
    # Data to be plotted
    cpu = df["Core Processors"]

    # Creating a histogram
    plt.figure(figsize=(8, 5))
    plt.hist(cpu, bins=8, color="skyblue", edgecolor="black")
    plt.xlabel("Core Processors")
    plt.ylabel("Frequency")
    plt.title("Core Processors Distribution")

    # Displaying the plot
    plt.show()


def pcamera_distribution(df):
    # Data to be plotted
    pc = df["Primary Camera (MP)"]

    # Creating a histogram
    plt.figure(figsize=(8, 5))
    plt.hist(pc, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Primary Camera (MP)")
    plt.ylabel("Frequency")
    plt.title("Primary Camera Quality Distribution")

    # Displaying the plot
    plt.show()


def fcamera_distribution(df):
    # Data to be plotted
    fc = df["Front Camera (MP)"]

    # Creating a histogram
    plt.figure(figsize=(8, 5))
    plt.hist(fc, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Front Camera (MP)")
    plt.ylabel("Frequency")
    plt.title("Front Camera Quality Distribution")

    # Displaying the plot
    plt.show()


def px_height_distribution(df):
    # Data to be plotted
    px_height = df["Pixel Resolution Height"]

    # Creating a histogram
    plt.figure(figsize=(8, 5))
    plt.hist(px_height, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Pixel Resolution Height")
    plt.ylabel("Frequency")
    plt.title("Pixel Resolution Height Distribution")

    # Displaying the plot
    plt.show()


def px_width_distribution(df):
    # Data to be plotted
    px_width = df["Pixel Resolution Width"]

    # Creating a histogram
    plt.figure(figsize=(8, 5))
    plt.hist(px_width, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Pixel Resolution Width")
    plt.ylabel("Frequency")
    plt.title("Pixel Resolution Width Distribution")

    # Displaying the plot
    plt.show()


# Numerical Features Across Price Ranges


def int_memory(df):
    # Extract labels from 'Price Range'
    price = df.groupby(["Price Range"]).size()
    labels = price.index
    x = range(len(labels))

    # Creating a violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Price Range",
        y="Internal Memory (GB)",
        hue="Price Range",
        data=df,
        palette="muted",
        legend=False,
    ).set(xlabel=None)

    # Adding labels and title
    plt.xticks(x, ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"])
    plt.ylabel("Internal Memory (GB)")
    plt.title("Internal Memory Distribution by Price Range")

    # Displaying the plot
    plt.show()


def ram(df):
    # Extract labels from 'Price Range'
    price = df.groupby(["Price Range"]).size()
    labels = price.index
    x = range(len(labels))

    # Creating a violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Price Range",
        y="RAM (MB)",
        hue="Price Range",
        data=df,
        palette="muted",
        legend=False,
    ).set(xlabel=None)

    # Adding labels and title
    plt.xticks(x, ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"])
    plt.ylabel("RAM (MB)")
    plt.title("RAM Distribution by Price Range")

    # Displaying the plot
    plt.show()


def battery(df):
    # Extract labels from 'Price Range'
    price = df.groupby(["Price Range"]).size()
    labels = price.index
    x = range(len(labels))

    # Creating a violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Price Range",
        y="Battery Power",
        hue="Price Range",
        data=df,
        palette="muted",
        legend=False,
    ).set(xlabel=None)

    # Adding labels and title
    plt.xticks(x, ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"])
    plt.ylabel("Battery Power")
    plt.title("Battery Power Distribution by Price Range")

    # Displaying the plot
    plt.show()


def clock_speed(df):
    # Extract labels from 'Price Range'
    price = df.groupby(["Price Range"]).size()
    labels = price.index
    x = range(len(labels))

    # Creating a violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Price Range",
        y="Clock Speed",
        hue="Price Range",
        data=df,
        palette="muted",
        legend=False,
    ).set(xlabel=None)

    # Adding labels and title
    plt.xticks(x, ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"])
    plt.ylabel("Core Processors")
    plt.title("Clock Speed Distribution by Price Range")

    # Displaying the plot
    plt.show()


def cpu(df):
    # Extract labels from 'Price Range'
    price = df.groupby(["Price Range"]).size()
    labels = price.index
    x = range(len(labels))

    # Creating a violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Price Range",
        y="Core Processors",
        hue="Price Range",
        data=df,
        palette="muted",
        legend=False,
    ).set(xlabel=None)

    # Adding labels and title
    plt.xticks(x, ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"])
    plt.ylabel("Core Processors")
    plt.title("Number of Core Processors Distribution by Price Range")

    # Displaying the plot
    plt.show()


def primary_camera(df):
    # Extract labels from 'Price Range'
    price = df.groupby(["Price Range"]).size()
    labels = price.index
    x = range(len(labels))

    # Creating a violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Price Range",
        y="Primary Camera (MP)",
        hue="Price Range",
        data=df,
        palette="muted",
        legend=False,
    ).set(xlabel=None)

    # Adding labels and title
    plt.xticks(x, ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"])
    plt.ylabel("Primary Camera (MP)")
    plt.title("Primary Camera Quality Distribution by Price Range")

    # Displaying the plot
    plt.show()


def front_camera(df):
    # Extract labels from 'Price Range'
    price = df.groupby(["Price Range"]).size()
    labels = price.index
    x = range(len(labels))

    # Creating a violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Price Range",
        y="Front Camera (MP)",
        hue="Price Range",
        data=df,
        palette="muted",
        legend=False,
    ).set(xlabel=None)

    # Adding labels and title
    plt.xticks(x, ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"])
    plt.ylabel("Front Camera (MP)")
    plt.title("Front Camera Quality Distribution by Price Range")

    # Displaying the plot
    plt.show()


def pixel_height(df):
    # Extract labels from 'Price Range'
    price = df.groupby(["Price Range"]).size()
    labels = price.index
    x = range(len(labels))

    # Creating a violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Price Range",
        y="Pixel Resolution Height",
        hue="Price Range",
        data=df,
        palette="muted",
        legend=False,
    ).set(xlabel=None)

    # Adding labels and title
    plt.xticks(x, ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"])
    plt.ylabel("Pixel Resolution Height")
    plt.title("Pixel Resolution Height Distribution by Price Range")

    # Displaying the plot
    plt.show()


def pixel_width(df):
    # Extract labels from 'Price Range'
    price = df.groupby(["Price Range"]).size()
    labels = price.index
    x = range(len(labels))

    # Creating a violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Price Range",
        y="Pixel Resolution Width",
        hue="Price Range",
        data=df,
        palette="muted",
        legend=False,
    ).set(xlabel=None)

    # Adding labels and title
    plt.xticks(x, ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"])
    plt.ylabel("Pixel Resolution Width")
    plt.title("Pixel Resolution Width Distribution by Price Range")

    # Displaying the plot
    plt.show()


# Categorical Features Across Price Ranges


def price_4g(df):
    # Grouping the DataFrame based on 'Price Range' and '4G', then calculate the count of each group
    price_4g = df.groupby(["Price Range", "4G"]).size().unstack()
    labels = price_4g.index
    x = range(len(labels))

    # Setting the width of each cluster
    cluster_width = 0.4
    x1 = np.arange(len(labels)) - cluster_width / 2
    x2 = np.arange(len(labels)) + cluster_width / 2

    # Creating a bar plot for each group
    plt.bar(x1, price_4g[1], cluster_width, label="4G", color="cornflowerblue")
    plt.bar(x2, price_4g[0], cluster_width, label="Not 4G", color="sandybrown")

    # Adding labels and title
    plt.xticks(x, ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"])
    plt.ylabel("Number of Mobile Phones")
    plt.title("4G Feature Distribution by Price Range")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Displaying the plot
    plt.show()


def price_3g(df):
    # Grouping the DataFrame based on 'Price Range' and '3G', then calculate the count of each group
    price_3g = df.groupby(["Price Range", "3G"]).size().unstack()
    labels = price_3g.index
    x = range(len(labels))

    # Setting the width of each cluster
    cluster_width = 0.4
    x1 = np.arange(len(labels)) - cluster_width / 2
    x2 = np.arange(len(labels)) + cluster_width / 2

    # Creating a bar plot for each group
    plt.bar(x1, price_3g[1], cluster_width, label="3G", color="cornflowerblue")
    plt.bar(x2, price_3g[0], cluster_width, label="Not 3G", color="sandybrown")

    # Adding labels and title
    plt.xticks(x, ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"])
    plt.ylabel("Number of Mobile Phones")
    plt.title("3G Feature Distribution by Price Range")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Displaying the plot
    plt.show()


def dual_sim(df):
    # Grouping the DataFrame based on 'Price Range' and 'Dual Sim', then calculate the count of each group
    price_dualsim = df.groupby(["Price Range", "Dual Sim"]).size().unstack()
    labels = price_dualsim.index
    x = range(len(labels))

    # Setting the width of each cluster
    cluster_width = 0.4
    x1 = np.arange(len(labels)) - cluster_width / 2
    x2 = np.arange(len(labels)) + cluster_width / 2

    # Creating a bar plot for each group
    plt.bar(
        x1, price_dualsim[1], cluster_width, label="Dual Sim", color="cornflowerblue"
    )
    plt.bar(
        x2, price_dualsim[0], cluster_width, label="Not Dual Sim", color="sandybrown"
    )

    # Adding labels and title
    plt.xticks(x, ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"])
    plt.ylabel("Number of Mobile Phones")
    plt.title("Dual Sim Feature Distribution by Price Range")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Displaying the plot
    plt.show()


def wifi(df):
    # Grouping the DataFrame based on 'Price Range' and 'Wi-Fi', then calculate the count of each group
    price_wifi = df.groupby(["Price Range", "Wi-Fi"]).size().unstack()
    labels = price_wifi.index
    x = range(len(labels))

    # Setting the width of each cluster
    cluster_width = 0.4
    x1 = np.arange(len(labels)) - cluster_width / 2
    x2 = np.arange(len(labels)) + cluster_width / 2

    # Creating a bar plot for each group
    plt.bar(x1, price_wifi[1], cluster_width, label="Wi-Fi", color="cornflowerblue")
    plt.bar(x2, price_wifi[0], cluster_width, label="No Wi-Fi", color="sandybrown")

    # Adding labels and title
    plt.xticks(x, ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"])
    plt.ylabel("Number of Mobile Phones")
    plt.title("Wi-Fi Feature Distribution by Price Range")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Displaying the plot
    plt.show()


def bluetooth(df):
    # Grouping the DataFrame based on 'Price Range' and 'Bluetooth', then calculate the count of each group
    price_blue = df.groupby(["Price Range", "Bluetooth"]).size().unstack()
    labels = price_blue.index
    x = range(len(labels))

    # Setting the width of each cluster
    cluster_width = 0.4
    x1 = np.arange(len(labels)) - cluster_width / 2
    x2 = np.arange(len(labels)) + cluster_width / 2

    # Creating a bar plot for each group
    plt.bar(x1, price_blue[1], cluster_width, label="Bluetooth", color="cornflowerblue")
    plt.bar(x2, price_blue[0], cluster_width, label="No Bluetooth", color="sandybrown")

    # Adding labels and title
    plt.xticks(x, ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"])
    plt.ylabel("Number of Mobile Phones")
    plt.title("Bluetooth Feature Distribution by Price Range")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Displaying the plot
    plt.show()


def touch_screen(df):
    # Grouping the DataFrame based on 'Price Range' and 'Bluetooth', then calculate the count of each group
    price_touch = df.groupby(["Price Range", "Touch Screen"]).size().unstack()
    labels = price_touch.index
    x = range(len(labels))

    # Setting the width of each cluster
    cluster_width = 0.4
    x1 = np.arange(len(labels)) - cluster_width / 2
    x2 = np.arange(len(labels)) + cluster_width / 2

    # Creating a bar plot for each group
    plt.bar(
        x1, price_touch[1], cluster_width, label="Touch Screen", color="cornflowerblue"
    )
    plt.bar(
        x2, price_touch[0], cluster_width, label="Not Touch Screen", color="sandybrown"
    )

    # Adding labels and title
    plt.xticks(x, ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"])
    plt.ylabel("Number of Mobile Phones")
    plt.title("Touch Screen Feature Distribution by Price Range")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Displaying the plot
    plt.show()


if __name__ == "__main__":
    # Load the data using the function from load_data.py
    file_path = "../data/Mobile Price Classification.csv"
    df = read_csv_data(file_path)

    # Preprocess the data
    df = preprocess_data(df)

    # Execute all the EDA functions
    price_distribution(df)
    int_memory_distribution(df)
    ram_distribution(df)
    battery_distribution(df)
    clk_speed_distribution(df)
    cpu_distribution(df)
    pcamera_distribution(df)
    fcamera_distribution(df)
    px_height_distribution(df)
    px_width_distribution(df)
    int_memory(df)
    ram(df)
    battery(df)
    clock_speed(df)
    cpu(df)
    primary_camera(df)
    front_camera(df)
    pixel_height(df)
    pixel_width(df)
    price_4g(df)
    price_3g(df)
    dual_sim(df)
    wifi(df)
    bluetooth(df)
    touch_screen(df)
