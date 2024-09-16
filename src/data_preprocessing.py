import pandas as pd
from load_data import read_csv_data  # Import function from load_data.py


# Define preprocessing functions
def rename_columns(df):
    df.rename(
        columns={
            "battery_power": "Battery Power",
            "blue": "Bluetooth",
            "clock_speed": "Clock Speed",
            "dual_sim": "Dual Sim",
            "fc": "Front Camera (MP)",
            "four_g": "4G",
            "int_memory": "Internal Memory (GB)",
            "m_dep": "Mobile Depth (cm)",
            "mobile_wt": "Weight",
            "n_cores": "Core Processors",
            "pc": "Primary Camera (MP)",
            "px_height": "Pixel Resolution Height",
            "px_width": "Pixel Resolution Width",
            "ram": "RAM (MB)",
            "sc_h": "Screen Height",
            "sc_w": "Screen Weight",
            "talk_time": "Talk Time",
            "three_g": "3G",
            "touch_screen": "Touch Screen",
            "wifi": "Wi-Fi",
            "price_range": "Price Range",
        },
        inplace=True,
    )
    return df


# Function to preprocess data
def preprocess_data(df):
    df = rename_columns(df)
    return df


# Test data_preprocessing as standalone
if __name__ == "__main__":
    # Load data using the function from Load_data.py
    file_path = "../data/Mobile Price Classification.csv"
    df = read_csv_data(file_path)

    # Display missing values in each columns
    null_value = df.isnull().sum()
    print(f"Missing values in each column:\n{null_value}")

    # Preprocess the data
    df = preprocess_data(df)

    # Set the maximum number of columns to display
    pd.set_option("display.max_columns", None)

    # Display the head of the dataframe after preprocessing
    print("Data after preprocessing:")
    print(df.head())
