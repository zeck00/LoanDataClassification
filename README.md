
# Loan Data Classification Project

## Dataset Overview
The dataset contains information about loan applications, with the target variable being `loan_status`, which indicates whether a loan was approved (`1`) or rejected (`0`). It includes both categorical and numerical features describing personal, financial, and loan-related attributes of individuals.

### Features
| Column                          | Description                                   | Type       |
|--------------------------------|-----------------------------------------------|------------|
| person_age                     | Age of the person                            | Float      |
| person_gender                  | Gender of the person                         | Categorical|
| person_education               | Highest education level                      | Categorical|
| person_income                  | Annual income                                | Float      |
| person_emp_exp                 | Years of employment experience               | Integer    |
| person_home_ownership          | Home ownership status (e.g., rent, own)      | Categorical|
| loan_amnt                      | Loan amount requested                        | Float      |
| loan_intent                    | Purpose of the loan                          | Categorical|
| loan_int_rate                  | Loan interest rate                           | Float      |
| loan_percent_income            | Loan amount as a percentage of annual income | Float      |
| cb_person_cred_hist_length     | Length of credit history in years            | Float      |
| credit_score                   | Credit score of the person                   | Integer    |
| previous_loan_defaults_on_file | Indicator of previous loan defaults          | Categorical|
| loan_status (target variable)  | Loan approval status: 1 = approved; 0 = rejected | Integer    |

### Dataset Statistics
- Total rows: 45,000
- No missing values were found.
- Categorical variables were encoded using `LabelEncoder`.

## Preprocessing Steps
1. **Categorical Encoding**:
   - Used `LabelEncoder` for encoding categorical variables.
2. **Feature Standardization**:
   - Standardized numerical features (`person_age`, `person_income`, `loan_amnt`, etc.) using `StandardScaler`.
3. **Train-Test Split**:
   - Split the dataset into training (80%) and testing (20%) sets.

## Models
Two classification models were implemented and evaluated:

### 1. Random Forest Classifier
- **Description**:
  - An ensemble learning method that uses multiple decision trees to improve classification accuracy.
- **Results**:
  - **Precision**: 0.94 (class 0), 0.89 (class 1)
  - **Recall**: 0.97 (class 0), 0.78 (class 1)
  - **F1-Score**: 0.95 (class 0), 0.83 (class 1)
  - **Overall Accuracy**: 92.83%

### 2. Gradient Boosting Classifier
- **Description**:
  - A boosting method that builds an additive model by optimizing a loss function.
- **Results**:
  - **Precision**: 0.93 (class 0), 0.87 (class 1)
  - **Recall**: 0.97 (class 0), 0.75 (class 1)
  - **F1-Score**: 0.95 (class 0), 0.81 (class 1)
  - **Overall Accuracy**: 92.01%

## Feature Importance
- Random Forest feature importance analysis revealed the following most influential features:
  1. `credit_score`
  2. `person_income`
  3. `loan_amnt`
  4. `cb_person_cred_hist_length`

## Visualization
1. **Feature Importance Plot**:
   - A bar plot showing the relative importance of features in the Random Forest model.
2. **Correlation Matrix**:
   - A heatmap visualizing correlations among features to identify multicollinearity.

## Recommendations
- The Random Forest Classifier performed slightly better in terms of recall for the minority class (`1`), making it more suitable for scenarios where minimizing false negatives is crucial.
- Gradient Boosting Classifier could be tuned further to achieve better recall for the minority class.

## How to Run the Code
1. Install the required libraries:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   ```
2. Place the dataset (`loan_data.csv`) in the working directory.
3. Run the script to preprocess the data, train the models, and visualize the results.

## Future Work
- Tune hyperparameters for all models to improve performance.
- Explore additional models such as XGBoost or neural networks for further improvement.
- Address potential class imbalance through techniques like SMOTE or class weighting.
