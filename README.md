# Project Title: Mobile Price Forecasting

This project will leverage data analysis and Random Forest machine learning models to extract insights from the sales data of mobile phones from various companies, helping to predict optimal pricing strategies for mobile phones.


## Project Description

This project uses a mobile phone sales dataset from various companies, which includes data on mobile phones along with their features (such as battery power, RAM, internal memory, etc) and their price range. The objectives of this project are as follows:
- Determine which features significantly influence the price range of mobile phones.
- Develop a machine learning model to predict the price range of a mobile phone based on its features.

## Directory Structure

- `data/`: Contains the dataset used in this project.
- `notebooks/`: Jupyter Notebooks for data exploration.
- `results/`: Contains model results, including evaluation and trained models.
  - `evaluation/`: Model evaluation results like the confusion matrix, classification report, ROC curve, and feature importance.
  - `model/`: The trained model saved in `.joblib` format.
- `src/`: Source code for the project.
  - `load_data.py`: Function to load data.
  - `data_preprocessing.py`: Function for data preprocessing.
  - `exploratory_data_analysis.py`: For data exploration.
  - `statistical_tests.py`: For correlation analysis.
  - `model_no_tuning.py`: Script to train the model without hyperparameter tuning, followed by evaluation using a confusion matrix and classification report.
  - `model_with_tuning.py`: Script to train the model with hyperparameter tuning, followed by evaluation using a confusion matrix and classification report.
  - `model_evaluation.py`: Function to evaluate both types of models using ROC curve and feature importance.
  - `main.py`: The main script for running prediction and model evaluation.

## Prerequisites

Ensure you have installed all required dependencies. You can install the dependencies by running:

```bash
pip install -r requirements.txt
```

## How to Run

To run the **Mobile Price Forecasting** project, follow these steps:

### 1. Clone the Repository
First, clone the repository to your local machine using the following command:
```bash
git clone https://github.com/m1racle31/Mobile-Price-Forecasting
cd Mobile-Price-Forecasting
```

### 2. Install Dependencies
Ensure all required dependencies are installed. This project uses Python, and the dependencies are listed in the `requirements.txt` file.

Install them using:
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset
Make sure the dataset is located in the `data/` folder. If the dataset is not available in the repository, download the **Mobile Price Classification** dataset and place it inside the `data/` directory.

### 4. Run the Preprocessing and Model Evaluation

You can execute the main script to load the data, preprocess it, load the trained model, and evaluate its performance. Simply run the following command:

```bash
python src/main.py
```

This script will:
- Load the data from the data/ folder.
- Preprocess the data
- Load the pre-trained Random Forest models (both before and after hyperparameter tuning) from results/model/.
- Make predictions on the test data.
- Evaluate the models' performance by calculating accuracy, generating the confusion matrix, plotting the ROC Curve, and displaying Feature Importance.

### 5. Additional Exploration with Notebooks (Optional)
If you want to explore the data or run specific analyses interactively, you can use the Jupyter Notebooks provided in the `notebooks/` folder.

To launch Jupyter Notebook:
```bash
jupyter notebook
```
Open any `.ipynb` files from the `notebooks/` directory for further exploration or analysis.

### 6. Output
Once the script completes:
- Model evaluation outputs such as the confusion matrix, classification report, ROC curve, and feature importance will be displayed.
- Both trained models will be saved in the results/model/ folder when running the model_no_tuning.py and model_with_tuning.py files, respectively.

## Code Explanation
This section provides a brief overview of the main source code files and their key functionalities.

- `src/load_data.py`: Contains functions for loading and reading the dataset.
  - `read_csv_data(file_path)`: Reads the CSV file from the given file path and returns a pandas DataFrame.

- `src/data_preprocessing.py`: Includes functions for preprocessing the data.
  - `preprocess_data(df)`: Handles missing values, feature engineering, and transforms the dataset to prepare it for modeling.

- `src/exploratory_data_analysis.py`: Functions used for performing exploratory data analysis (EDA) on the dataset.

- `src/statistical_tests.py`: Contains code for statistical analysis and correlation testing.
  - `calculate_correlation_matrix(df)`: Conducts correlation analysis between features and the target variable to understand relationships.

- `src/model_no_tuning.py`: Contains the code for building and training the prediction model (without hyperparameter tuning), and then saving the model.
- `src/model_with_tuning.py`: Contains the code for building and training the prediction model (with hyperparameter tuning), and then saving the model.

- `src/model_evaluation.py`: Contains code for evaluating saved models using ROC Curve and Feature Importance.
  - `plot_roc_curves(y_true, y_pred_prob_dict, save_path)`: Generates and plots ROC Curves to evaluate the model's performance.
  - `plot_feature_importance(model_path, preprocessed_df, selected_features, categorical_cols, save_path)`: Plots the Feature Importance of the model to understand the impact of each feature.

- `src/main.py`: The main script for running the prediction and evaluation process.
  - `main()`: Orchestrates the full workflow including data loading, preprocessing, loading the pre-trained model, predicting on the test set, and evaluating the model.

- `results/`: This folder contains the following subdirectories:
  - `evaluation/`: Stores the evaluation results such as confusion matrix, classification reports, ROC curve, and feature importance.
  - `model/`: Contains both trained models (with and without hyperparameter tuning) saved in `.joblib` format.

## Contact

For any questions or further assistance, feel free to reach out to me:

- **Email**: [yoshuaaugusta31@gmail.com](mailto:yoshuaaugusta31@gmail.com)
- **LinkedIn**: [Yoshua Augusta](https://www.linkedin.com/in/yoshua-augusta/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.