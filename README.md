# Kaggle Titanic Survival Prediction 🚢

## Overview

This project tackles the classic Kaggle competition: "[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)". The goal is to build a machine learning model that predicts whether a passenger on the RMS Titanic survived the infamous shipwreck based on available passenger data.

This repository contains the code, analysis, and documentation for exploring the dataset, engineering relevant features, training various classification models, and generating predictions for submission.

## The Challenge

The sinking of the RMS Titanic on April 15, 1912, is one of the most well-known maritime disasters. Despite being considered "unsinkable," the ship struck an iceberg on its maiden voyage, leading to the deaths of 1502 out of 2224 passengers and crew due to insufficient lifeboats.

While survival involved some luck, certain groups of people were more likely to survive than others. This project aims to answer the question: **“what sorts of people were more likely to survive?”** by using machine learning techniques on passenger data like name, age, gender, socio-economic class, etc.

## Dataset

The data is provided by Kaggle and split into two files:

1.  **`train.csv`**: Contains details for a subset of passengers (891) and includes the ground truth – whether they survived (`Survived` column: 0 = No, 1 = Yes). This dataset is used for training the model.
2.  **`test.csv`**: Contains similar details for another set of passengers (418) but *without* the `Survived` column. This dataset is used to generate predictions.

Key features include:
* `PassengerId`: Unique ID for each passenger.
* `Survived`: Survival status (0 = No, 1 = Yes) - Target variable (in `train.csv` only).
* `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) - Proxy for socio-economic status.
* `Sex`: Passenger's sex (male, female).
* `Age`: Passenger's age in years.
* `SibSp`: Number of siblings/spouses aboard the Titanic.
* `Parch`: Number of parents/children aboard the Titanic.
* `Ticket`: Ticket number.
* `Fare`: Passenger fare.
* `Cabin`: Cabin number.
* `Embarked`: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

You can find the data and more details on the [Kaggle Data Page](https://www.kaggle.com/competitions/titanic/data).

## Methodology

The project follows these general steps:

1.  **Data Loading & Initial Setup**: Load `train.csv` and `test.csv` into pandas DataFrames.
2.  **Exploratory Data Analysis (EDA)** 📊: Analyze variables, visualize distributions, identify correlations, and understand relationships between features and survival.
3.  **Data Cleaning**: Handle missing values (e.g., `Age`, `Embarked`, `Cabin`, `Fare`) using appropriate imputation strategies.
4.  **Feature Engineering** ✨:
    * Create new features from existing ones (e.g., `FamilySize`, `IsAlone`, extracting `Title` from `Name`).
    * Convert categorical features (`Sex`, `Embarked`, `Title`) into numerical representations suitable for machine learning models (e.g., one-hot encoding, label encoding).
    * Bin numerical features like `Age` and `Fare` if beneficial.
5.  **Model Selection & Training**:
    * Select appropriate classification algorithms (e.g., Logistic Regression, K-Nearest Neighbors, Support Vector Machines, Decision Trees, Random Forest, Gradient Boosting).
    * Train selected models on the processed training data.
    * Use techniques like cross-validation to evaluate model performance robustly and tune hyperparameters.
6.  **Prediction**: Use the best-performing trained model to predict survival outcomes for the passengers in the `test.csv` dataset.
7.  **Submission**: Format the predictions into a CSV file according to Kaggle's submission guidelines (`PassengerId`, `Survived`).

## Technologies Used

* **Language:** Python 3.x
* **Libraries:**
    * `pandas`: Data manipulation and analysis.
    * `numpy`: Numerical operations.
    * `matplotlib` & `seaborn`: Data visualization.
    * `scikit-learn`: Machine learning (preprocessing, models, evaluation, tuning).
    * `Jupyter Notebooks` / `Google Colab`: Interactive development and analysis environment.
