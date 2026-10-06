# 💳 UPI Fraud Detection System

A Machine Learning-based web application designed to detect potentially fraudulent UPI transactions using transaction-related features and a Random Forest classification model.

The project combines **Python, Machine Learning, Data Processing, and a web-based dashboard** to provide a simple interface where transaction details can be entered and evaluated for possible fraud.

---

## 📌 Project Overview

UPI has become one of the most widely used digital payment methods in India. With the increasing number of digital transactions, the possibility of fraudulent transactions has also increased.

This project aims to build a system that can analyze UPI transaction data and classify a transaction as either:

* ✅ **Legitimate Transaction**
* 🚨 **Fraudulent Transaction**

The system uses a **Random Forest Classifier** trained on a UPI transaction dataset. A web interface is provided so that users can interact with the trained model without directly working with Python code.

---

## 🎯 Objectives

The main objectives of this project are:

* To understand the characteristics of UPI transactions.
* To preprocess and analyze transaction data.
* To identify patterns associated with fraudulent transactions.
* To train a Machine Learning classification model.
* To evaluate the performance of the trained model.
* To create a simple and user-friendly fraud detection interface.
* To demonstrate how Machine Learning can be applied to digital payment security.

---

## ✨ Key Features

* 📊 Transaction data analysis
* 🧹 Data preprocessing and preparation
* 🤖 Random Forest classification
* 🔍 Fraud/legitimate transaction prediction
* 🌐 Interactive web interface
* 📈 Model performance evaluation
* 📋 Transaction-based prediction
* ⚡ Simple and easy-to-use interface

---

## 🛠️ Technologies Used

| Technology        | Purpose                               |
| ----------------- | ------------------------------------- |
| **Python**        | Main programming language             |
| **Pandas**        | Data manipulation and preprocessing   |
| **NumPy**         | Numerical operations                  |
| **Scikit-learn**  | Machine Learning model and evaluation |
| **Matplotlib**    | Data visualization                    |
| **Random Forest** | Classification algorithm              |
| **VS Code**       | Development environment               |
| **Streamlit**     | Web application/dashboard             |

---

## 📂 Dataset

The project uses a UPI transaction dataset containing **26,393 records and 65 columns**.

The dataset contains different transaction-related attributes that can be used to identify patterns in fraudulent transactions.

The target variable used for classification is:

```text
is_fraud
```

It represents whether a particular transaction is classified as fraudulent or legitimate.

### Dataset Distribution

| Transaction Type |      Count |
| ---------------- | ---------: |
| Legitimate       |     21,848 |
| Fraudulent       |      4,545 |
| **Total**        | **26,393** |

The dataset contains a mixture of transaction, account, device, location, and verification-related features.

---

# 🔄 Project Workflow

The overall workflow of the project is:

```text
        UPI Transaction Dataset
                  ↓
          Data Preprocessing
                  ↓
       Feature Selection / Encoding
                  ↓
          Train-Test Split
                  ↓
        Random Forest Classifier
                  ↓
         Model Evaluation
                  ↓
        Save / Use Trained Model
                  ↓
          Web Application
                  ↓
       Enter Transaction Details
                  ↓
          Fraud Prediction
```

---

## 1️⃣ Data Collection

The first step is to load the UPI transaction dataset using Pandas.

The dataset is inspected to understand:

* Number of rows and columns
* Data types
* Missing values
* Unique values
* Distribution of the target variable
* Important transaction-related features

Example:

```python
import pandas as pd

df = pd.read_csv("fraud_dataset.csv")

print(df.shape)
print(df.head())
```

---

## 2️⃣ Data Preprocessing

Raw datasets may contain categorical values, missing values, irrelevant columns, or information that cannot be directly processed by a Machine Learning algorithm.

Therefore, preprocessing is performed before training the model.

The preprocessing stage includes tasks such as:

* Handling missing values
* Converting categorical variables into numerical form
* Selecting relevant features
* Removing unnecessary information
* Preparing the target variable
* Splitting the data into training and testing sets

---

## 3️⃣ Feature Selection

The dataset contains multiple transaction-related features.

Features can represent information such as:

* Transaction amount
* Transaction type
* Device information
* Location
* Account information
* Verification status
* Transaction frequency
* Other behavioral or transaction attributes

The selected features are provided to the Machine Learning model to identify patterns related to fraudulent transactions.

---

# 🤖 Machine Learning Model

## Random Forest Classifier

The project uses a **Random Forest Classifier** for fraud classification.

Random Forest is an ensemble Machine Learning algorithm that combines multiple decision trees to make a final prediction.

Instead of relying on a single decision tree, Random Forest creates several trees and combines their predictions.

### Why Random Forest?

Random Forest was selected because it:

* Works well with classification problems.
* Can handle many features.
* Can capture non-linear relationships.
* Is relatively robust to noise.
* Provides feature importance information.
* Generally performs well on structured/tabular datasets.

---

## 🧠 How the Model Works

The model learns patterns from the training transactions.

For a new transaction:

```text
New Transaction
       ↓
Feature Processing
       ↓
Random Forest Model
       ↓
Multiple Decision Trees
       ↓
Combined Prediction
       ↓
Fraud / Legitimate
```

The final classification is based on the combined output of the individual decision trees.

---

# 📊 Model Evaluation

The trained model is evaluated using common classification metrics.

### Accuracy

Measures the overall percentage of correctly classified transactions.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Precision tells us how many transactions predicted as fraud were actually fraudulent.

```text
Precision = True Positives / (True Positives + False Positives)
```

### Recall

Recall measures how many actual fraudulent transactions were successfully detected.

```text
Recall = True Positives / (True Positives + False Negatives)
```

### F1-Score

F1-score combines precision and recall into a single metric.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

# ⚠️ Important Model Observation

During the initial model evaluation, the Random Forest produced an unusually perfect result, with:

* Accuracy = **1.00**
* Precision = **1.00**
* Recall = **1.00**
* F1-score = **1.00**
* Zero errors in the confusion matrix

While this initially appeared to indicate excellent performance, a perfect score on a real-world-style fraud dataset can also be a warning sign.

Therefore, the model was investigated further instead of treating the result as automatically reliable.

Feature importance analysis was performed to identify which features were contributing most strongly to the predictions.

This investigation highlighted the possibility of **data leakage**.

---

# 🔎 Data Leakage

Data leakage occurs when information that would not realistically be available at the time of prediction is accidentally provided to the Machine Learning model.

For example, a feature that is generated **after a transaction has already been verified or classified** could give the model information about the outcome itself.

This can cause the model to achieve unrealistically high performance during testing.

Therefore, features such as verification/status-related attributes need to be carefully examined before using the model in a real-world fraud detection system.

### Key Learning

> A high accuracy score does not always mean that a Machine Learning model is reliable.

Understanding the dataset and checking for leakage is an important part of Machine Learning model development.

---

# 🌐 Web Application

A web-based interface was developed so that the project can be used without directly interacting with the Python source code.

The application allows the user to provide transaction-related information and receive a prediction from the trained model.

### Application Flow

```text
User enters transaction details
              ↓
       Input preprocessing
              ↓
      Trained ML model
              ↓
       Prediction generated
              ↓
   Fraud / Legitimate result
```

The dashboard makes the project more interactive and demonstrates how a Machine Learning model can be integrated into an application.

---

# 🖥️ Running the Project

## Step 1 — Clone the Repository

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd UPI-Fraud-Detection
```

---

## Step 2 — Install Dependencies

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib streamlit
```

If a `requirements.txt` file is provided, use:

```bash
pip install -r requirements.txt
```

---

## Step 3 — Check the Dataset

Make sure the dataset file is present in the appropriate project directory.

Example:

```text
fraud_dataset.csv
```

---

## Step 4 — Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📁 Project Structure

A typical project structure is:

```text
UPI-Fraud-Detection/
│
├── app.py
├── fraud_dataset.csv
├── model/
│   └── trained_model.pkl
│
├── notebooks/
│   └── analysis.ipynb
│
├── visualizations/
│   └── graphs/
│
├── requirements.txt
└── README.md
```

> The exact files and folders may vary depending on the final version of the project.

---

# 📈 Data Visualizations

Data visualization was used during the analysis stage to understand the dataset and identify patterns.

The project includes visual analysis such as:

* Fraud vs legitimate transaction distribution
* Feature distributions
* Transaction patterns
* Model-related analysis
* Feature importance

These visualizations help understand the dataset before building the Machine Learning model.

---

# ✅ Advantages

* Easy-to-use interface
* Uses Machine Learning for automated classification
* Can process multiple transaction-related features
* Random Forest provides good performance on structured data
* Helps understand patterns in fraudulent transactions
* Can be extended for more advanced fraud detection systems
* Demonstrates the complete workflow from dataset to deployed application

---

# ⚠️ Limitations

* The model depends heavily on the quality of the dataset.
* Dataset imbalance can affect fraud classification.
* The current dataset may not represent every type of real-world UPI fraud.
* A model showing perfect test performance may indicate data leakage.
* Real-time banking data is not connected to the application.
* Fraud patterns can change over time.
* The current project is intended as an academic/educational implementation rather than a production banking security system.

---

# 🚀 Future Scope

The project can be further improved by:

* Using a larger and more diverse transaction dataset.
* Removing or carefully controlling features that may cause data leakage.
* Comparing multiple algorithms such as Logistic Regression, Decision Tree, XGBoost, and Support Vector Machines.
* Handling class imbalance using suitable techniques.
* Implementing real-time transaction monitoring.
* Adding anomaly detection.
* Introducing explainable predictions so users can understand why a transaction was flagged.
* Connecting the system to a secure transaction API.
* Deploying the application on a cloud platform.
* Continuously retraining the model with newer fraud patterns.

---

# 📚 Key Learnings

Through this project, the following concepts were explored:

* Data preprocessing
* Exploratory Data Analysis
* Feature selection
* Classification
* Random Forest
* Train-test splitting
* Model evaluation
* Confusion matrix
* Precision, Recall and F1-score
* Feature importance
* Data leakage
* Machine Learning application development
* Web application integration

One of the most important learnings was that **Machine Learning performance must be interpreted carefully**. A model achieving 100% accuracy should be investigated rather than automatically considered perfect.

---

# 🎓 Project Type

**Mini Project — Machine Learning / Python**

**Domain:** Digital Payments & Fraud Detection

**Application:** UPI Transaction Fraud Classification

---

# 👩‍💻 Author

**Aarushi Shah**
**Priya Sahani**
**Amaan Shaikh**

Artificial Intelligence & Machine Learning
Thakur College of Engineering & Technology (TCET)

---

# ⭐ Conclusion

The UPI Fraud Detection System demonstrates how Machine Learning can be used to classify digital payment transactions based on transaction-related patterns.

The project covers the complete development process — from **data loading and preprocessing to model training, evaluation, investigation of data leakage, and integration with a web application**.

Although the current implementation is an academic project, it provides a foundation for developing more advanced and reliable fraud detection systems in the future.

---

## 📌 Disclaimer

This project is developed for **educational and academic purposes**. It is not intended to replace real-world banking fraud detection or security systems. Real financial systems require significantly larger datasets, secure infrastructure, continuous monitoring, rigorous validation, and domain-specific security measures.
