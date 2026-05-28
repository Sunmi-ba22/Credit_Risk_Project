# Credit Risk Prediction Project

## Overview

This project focuses on predicting loan default risk using Machine Learning techniques. The objective is to identify borrowers who are likely to default on their loans based on demographic, financial, and credit-related information.

The project covers the complete machine learning workflow, including:

* Exploratory Data Analysis (EDA)
* Data Cleaning and Preprocessing
* Feature Engineering
* Handling Class Imbalance
* Model Building
* Model Evaluation
* Performance Comparison
* Business Insights

The final goal is to help financial institutions make better lending decisions by minimizing credit risk while reducing unnecessary loan rejections.

---

# Problem Statement

Financial institutions face significant losses when borrowers fail to repay loans. Traditional credit assessment methods may not effectively capture complex borrower behavior patterns.

This project aims to build predictive machine learning models capable of:

* Identifying high-risk borrowers
* Reducing loan default rates
* Improving risk assessment accuracy
* Supporting data-driven lending decisions

---

# Dataset Description

The dataset contains borrower demographic information, financial indicators, loan characteristics, and historical credit behavior.

## Features

| Feature              | Description                                   |
| -------------------- | --------------------------------------------- |
| Age                  | Age of borrower                               |
| Income               | Borrower's annual income                      |
| Home_Ownership       | Home ownership status                         |
| Employment_Years     | Years of employment                           |
| Loan_Purpose         | Purpose of the loan                           |
| Loan_Risk_Grade      | Assigned loan risk category                   |
| Loan_Amount          | Total loan amount                             |
| Interest_Rate        | Interest rate on loan                         |
| Loan_Default_Status  | Target variable (0 = No Default, 1 = Default) |
| Loan_Income_Ratio    | Ratio of loan amount to income                |
| Previous_Default     | Whether borrower defaulted previously         |
| Credit_History_Years | Length of credit history                      |

---

# Project Workflow

## 1. Data Cleaning & Preprocessing

The following preprocessing steps were performed:

* Checked for missing values
* Removed duplicates
* Verified data types
* Encoded categorical variables
* Scaled numerical features where necessary
* Split data into training and testing sets

---

## 2. Exploratory Data Analysis (EDA)

EDA was conducted to understand borrower behavior patterns and identify variables associated with loan default.

### Key Findings

* The dataset showed moderate class imbalance:

  * Non-default: 78.1%
  * Default: 21.9%

* Loan-to-income ratio and interest rate showed the strongest positive correlation with default risk.

* Higher-income borrowers were slightly less likely to default.

* Borrowers with previous defaults exhibited higher default tendencies.

### Visualizations Performed

* Histograms
* Boxplots
* Correlation Heatmaps
* ROC Curves
* Precision-Recall Curves
* Confusion Matrices
* Model Comparison Charts

---

# Machine Learning Models

The following classification models were implemented and evaluated:

## 1. Logistic Regression

### Performance

* AUC-ROC: 0.850
* F1-Score: 0.592
* Precision: 0.479
* Recall: 0.776

### Observations

* Strong baseline model
* High recall for identifying defaulters
* Lower precision caused many false positives
* Limited by linear decision boundaries

---

## 2. Random Forest

### Performance

* AUC-ROC: 0.933
* F1-Score: 0.820
* Precision: 0.973
* Recall: 0.709

### Observations

* Excellent classification performance
* Very high precision
* Lower recall than Logistic Regression
* Captured non-linear relationships effectively

---

## 3. Gradient Boosting

### Performance

* AUC-ROC: 0.945
* F1-Score: 0.835
* Precision: 0.972
* Recall: 0.731

### Observations

* Best overall model
* Strong balance between precision and recall
* Highest discrimination capability
* Most effective at identifying risky borrowers

---

# Model Comparison

| Model               | AUC-ROC | Precision | Recall | F1-Score |
| ------------------- | ------- | --------- | ------ | -------- |
| Logistic Regression | 0.850   | 0.479     | 0.776  | 0.592    |
| Random Forest       | 0.933   | 0.973     | 0.709  | 0.820    |
| Gradient Boosting   | 0.945   | 0.972     | 0.731  | 0.835    |

---

# Final Model Selection

Gradient Boosting was selected as the final model because it achieved:

* The highest AUC-ROC score
* The best F1-score
* Strong precision-recall balance
* Excellent overall classification performance

This makes it the most suitable model for predicting loan defaults while minimizing both false positives and false negatives.

---

# Technologies Used

## Programming Language

* Python

## Libraries

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

---

# Project Structure

```bash
credit-risk-prediction/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── credit_risk_analysis.ipynb
│
├── images/
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   └── precision_recall_curve.png
│
├── models/
│   └── trained_model.pkl
│
├── requirements.txt
├── README.md
└── app.py
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Sunmi-ba22/credit-risk-prediction.git
```

## Navigate into Project Folder

```bash
cd credit-risk-prediction
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

Run the notebook:

```bash
jupyter notebook
```

Or run the Python script:

```bash
python app.py
```

---

# Business Impact

This project demonstrates how machine learning can support financial institutions by:

* Reducing lending risk
* Improving borrower assessment
* Supporting automated decision-making
* Minimizing financial losses from defaults
* Enhancing operational efficiency

---

# Future Improvements

Potential future enhancements include:

* Hyperparameter tuning

* Cross-validation optimization

* SMOTE for class imbalance handling

* Model deployment using Streamlit

* Real-time prediction API

* Explainable AI techniques (SHAP values)

---

# Key Learning Outcomes

Through this project, the following skills were demonstrated:

* Exploratory Data Analysis
* Feature Engineering
* Classification Modeling
* Model Evaluation
* Data Visualization
* Credit Risk Analysis
* Machine Learning Workflow

---

# Author

## Sunmisola Lawal

Aspiring Data Scientist passionate about Machine Learning, Analytics, and solving real-world business problems with data.

### Connect With Me

* LinkedIn: [Add Your LinkedIn]
* GitHub: [Add Your GitHub]
* Portfolio: [Add Portfolio Link]

---

# License

This project is licensed under the MIT License.
