import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from load_data import read_csv_data  # Import function from load_data.py
from data_preprocessing import preprocess_data  # Import the preprocessing function


# Define function to plot and save combined ROC Curves
def plot_roc_curves(y_true, y_pred_prob_dict, save_path):
    plt.figure(figsize=(12, 6))

    for model_name, y_pred_prob in y_pred_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = %0.2f)" % roc_auc)

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    # Adjust layout and save plot
    plt.tight_layout()  # Adjust layout to ensure everything fits
    plt.savefig(
        save_path, bbox_inches="tight"
    )  # Save plot as PNG file with tight bounding box
    plt.close()


# Define function to plot and save Feature Importance
def plot_feature_importance(
    model_path, preprocessed_df, selected_features, categorical_cols, save_path
):
    # Load the model
    model = joblib.load(model_path)

    # Prepare the feature set by encoding categorical variables
    X = pd.get_dummies(preprocessed_df[selected_features], columns=categorical_cols)

    # Get feature importances from the loaded model
    feature_importances = model.feature_importances_

    # Create a DataFrame to store feature importances
    feature_importance_df = pd.DataFrame(
        {"Feature": X.columns, "Importance": feature_importances}
    )

    # Sort the DataFrame by importance values in descending order
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )
    feature_importance_df = feature_importance_df.reset_index(drop=True)

    # Print the Feature Importance dataframe
    print(feature_importance_df)

    # Plot feature importances
    plt.figure(figsize=(9, 6))
    plt.barh(
        feature_importance_df["Feature"],
        feature_importance_df["Importance"],
        color="skyblue",
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top

    # Adjust layout and save plot
    plt.tight_layout()  # Adjust layout to ensure everything fits
    plt.savefig(
        save_path, bbox_inches="tight"
    )  # Save plot as PNG file with tight bounding box
    plt.close()


if __name__ == "__main__":
    # Load and preprocess the data
    file_path = "../data/Mobile Price Classification.csv"
    df = read_csv_data(file_path)
    df = preprocess_data(df)

    # Selected features and categorical variables
    categorical_cols = ["Bluetooth", "Dual Sim", "4G"]
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

    # Load models
    model_no_tuning_path = "../results/model/random_forest_model (no tuning).joblib"
    model_with_tuning_path = "../results/model/random_forest_model (with tuning).joblib"
    model_no_tuning = joblib.load(model_no_tuning_path)
    model_with_tuning = joblib.load(model_with_tuning_path)

    # Predict probabilities
    y_pred_prob_with_tuning = model_no_tuning.predict_proba(X_test)[:, 1]
    y_pred_prob_no_tuning = model_with_tuning.predict_proba(X_test)[:, 1]

    # Define paths to save plots
    roc_curve_path = "../results/evaluation/roc_curve.png"
    feature_importance_path = "../results/evaluation/feature_importance.png"

    # Plot and save combined ROC Curve for both models
    y_pred_prob_dict = {
        "Model Before Tuning": y_pred_prob_no_tuning,
        "Model After Tuning": y_pred_prob_with_tuning,
    }
    plot_roc_curves(y_test, y_pred_prob_dict, roc_curve_path)

    # Plot and save Feature Importance for the tuned model
    plot_feature_importance(
        model_with_tuning_path,
        df,
        selected_features,
        categorical_cols,
        feature_importance_path,
    )
