import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from load_data import read_csv_data  # Import function from load_data.py
from data_preprocessing import preprocess_data  # Import the preprocessing function


def load_model(model_path):
    # Load saved model
    model = joblib.load(model_path)
    return model


def evaluate_model(y_test, y_pred, labels, model_name):
    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Accuracy for {model_name}:", accuracy)
    print(f"Confusion Matrix for {model_name}:\n", conf_matrix)
    print(f"Classification Report for {model_name}:\n", class_report)

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
    plt.title(f"Mobile Price Classification ({model_name})", fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_roc_curves(y_true, y_pred_prob_dict):
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
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    model, preprocessed_df, selected_features, categorical_cols
):
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
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def main():
    # Load and preprocess the data
    file_path = "../data/Mobile Price Classification.csv"
    df = read_csv_data(file_path)
    df = preprocess_data(df)

    # Categorical variables and selected features
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
    X = pd.get_dummies(df[selected_features], columns=categorical_cols)
    y = df["Price Range"]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load the pre-trained models
    rf_clf_no_tuning = load_model(
        "../results/model/random_forest_model (no tuning).joblib"
    )
    rf_clf_with_tuning = load_model(
        "../results/model/random_forest_model (with tuning).joblib"
    )

    # Predictions for both models
    y_pred_no_tuning = rf_clf_no_tuning.predict(X_test)
    y_pred_with_tuning = rf_clf_with_tuning.predict(X_test)

    # Evaluate both models
    labels = ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"]
    evaluate_model(y_test, y_pred_no_tuning, labels, "no tuning")
    evaluate_model(y_test, y_pred_with_tuning, labels, "with tuning")

    # Predict probabilities for ROC Curve
    y_pred_prob_no_tuning = rf_clf_no_tuning.predict_proba(X_test)[:, 1]
    y_pred_prob_with_tuning = rf_clf_with_tuning.predict_proba(X_test)[:, 1]
    y_pred_prob_dict = {
        "Model Before Tuning": y_pred_prob_no_tuning,
        "Model After Tuning": y_pred_prob_with_tuning,
    }
    plot_roc_curves(y_test, y_pred_prob_dict)

    # Plot Feature Importance for the tuned model
    plot_feature_importance(
        rf_clf_with_tuning,
        df,
        selected_features,
        categorical_cols,
    )


if __name__ == "__main__":
    main()
