# Predicting-Customer-Churn

This project aims to predict customer churn using machine learning models. Customer churn refers to when a customer stops doing business with a company. By predicting churn, businesses can take proactive steps to retain their customers.

---

## 📊 Dataset

The dataset used in this project is the **Telco Customer Churn** dataset.

- 📥 **Download it from Kaggle**:  
  [Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- 📝 **Description**:
  This dataset contains information about a telecom company's customers and whether they have churned. It includes fields like:
  - CustomerID
  - Gender
  - Senior Citizen
  - Partner and Dependents
  - Tenure
  - Phone and Internet service details
  - Contract and payment information
  - Monthly and Total charges
  - Churn (target variable)

---

## 📁 Project Structure

- `Predicting_Customer_Churn_using_Machine_Learning.ipynb`: Main Jupyter Notebook with all the code, visualizations, model training, and evaluation.
- (Optional) `requirements.txt`: List of dependencies to recreate the environment.

---

## 🚀 Features

- Exploratory Data Analysis (EDA)
- Data preprocessing: handling missing values, encoding, and scaling
- Multiple classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
- Model evaluation using:
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)

---

## 🧰 Technologies Used

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib & Seaborn
- Scikit-learn

---

## 📦 Installation & Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/customer-churn-ml.git
   cd customer-churn-ml

2. (Optional) Create a virtual environment:
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
3. Install dependencies:
   pip install -r requirements.txt
4. Run the notebook:
   jupyter notebook
5. Add the dataset:
   Download the dataset from Kaggle and place the WA_Fn-UseC_-Telco-Customer-Churn.csv file in the root project directory or update the path in the notebook.
