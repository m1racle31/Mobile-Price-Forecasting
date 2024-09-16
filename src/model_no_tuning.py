import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from load_data import read_csv_data  # Import function from load_data.py
from data_preprocessing import preprocess_data  # Import the preprocessing function


def main():
    # Load and preprocess the data
    file_path = "../data/Mobile Price Classification.csv"
    df = read_csv_data(file_path)
    df = preprocess_data(df)

    # Categorical variables that need to be encoded
    categorical_cols = ["Bluetooth", "Dual Sim", "4G"]

    # Selected features
    selected_features = [
        "Battery Power",
        "Bluetooth",
        "Clock Speed",
        "Core Processors",
        "Dual Sim",
        "Front Camera (MP)",
        "Primary Camera (MP)",
        "Internal Memory (GB)",
        "RAM (MB)",
        "4G",
        "Pixel Resolution Height",
        "Pixel Resolution Width",
    ]

    # Encode categorical variables using pd.get_dummies
    X = pd.get_dummies(df[selected_features], columns=categorical_cols)
    y = df["Price Range"]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train the model
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)

    # Save the model to a file
    os.makedirs("../results/model", exist_ok=True)
    model_filename = "../results/model/random_forest_model (no tuning).joblib"
    joblib.dump(rf_clf, model_filename)

    # Predict and evaluate the model performance
    y_pred = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Save evaluation results
    os.makedirs("../results/evaluation", exist_ok=True)

    # Define label names
    labels = ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"]

    # Save confusion matrix as a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Mobile Price Classification", fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("../results/evaluation/confusion_matrix (no tuning).png")
    plt.close()

    # Save classification report
    with open("../results/evaluation/classification_report (no tuning).txt", "w") as f:
        f.write(class_report)


if __name__ == "__main__":
    main()
