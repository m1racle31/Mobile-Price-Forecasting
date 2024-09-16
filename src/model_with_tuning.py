import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from load_data import read_csv_data  # Import function from load_data.py
from data_preprocessing import preprocess_data  # Import the preprocessing function


def main():
    # Load and preprocess the data
    file_path = (
        "../data/Mobile Price Classification.csv"  # Ganti dengan path yang sesuai
    )
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

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define hyperparameters to tune
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }

    # Initialize and perform Grid Search
    rf_clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
    )
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Predict with the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Save the model with best hyperparameters
    os.makedirs("../results/model", exist_ok=True)
    model_filename = "../results/model/random_forest_model (with tuning).joblib"
    joblib.dump(best_model, model_filename)

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
    plt.title("Mobile Price Classification (Hyperparameter Tuning)", fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("../results/evaluation/confusion_matrix (with tuning).png")
    plt.close()

    # Save classification report
    with open(
        "../results/evaluation/classification_report (with tuning).txt", "w"
    ) as f:
        f.write(class_report)


if __name__ == "__main__":
    main()
