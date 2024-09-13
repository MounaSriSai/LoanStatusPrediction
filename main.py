#!/usr/bin/env python
# coding: utf-8

# #### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import joblib
from tkinter import *
from tkinter import messagebox
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# #### Data Preprocessing Function

# In[2]:


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
     # EDA and Visualization
    print(df.head())
    print("Data Description:")
    print(df.describe())
    print(df.info())
    print('*****************************************************************************')
    print("\nCrosstab of Credit History vs Loan Status:")
    print(pd.crosstab(df['Credit_History'], df['Loan_Status'], margins=True))
    print('*****************************************************************************')
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    df.boxplot(column='ApplicantIncome')
    plt.title('Applicant Income Distribution')
    
    plt.subplot(1, 3, 2)
    df['ApplicantIncome'].hist(bins=20)
    plt.title('Applicant Income Histogram')
    
    plt.subplot(1, 3, 3)
    df['CoapplicantIncome'].hist(bins=20)
    plt.title('Coapplicant Income Histogram')
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    df.boxplot(column='ApplicantIncome', by='Education')
    plt.title('Applicant Income by Education')
    plt.show()
    
    plt.subplot(1, 2, 2)
    df['LoanAmount'].hist(bins=20)
    plt.title('Loan Amount Histogram')
    plt.show()
    
    
    print('*****************************************************************************')
    # Data Preprocessing
    print('Null values\n',df.isnull().sum())
    print('*****************************************************************************')
    df = df.dropna(subset=['Gender', 'Married', 'Dependents', 'LoanAmount', 'Loan_Amount_Term', 'Education', 'ApplicantIncome', 'CoapplicantIncome', 'Property_Area'])
    # Checking for missing Values
    print(df['Loan_Status'].value_counts())
    print('*****************************************************************************')
    print('Loan_ID Duplicate Values',df.duplicated(subset = ['Loan_ID']).sum())
    df = df.drop_duplicates(subset=['Loan_ID'])
    df['Credit_History'] = df['Credit_History'].fillna(0).astype(int)
    df['Self_Employed'] = df['Self_Employed'].fillna('No')
    df['Property_Area'] = df['Property_Area'].replace({'SEMIURBAN': 'Semiurban', 'URBAN': 'Urban', 'RURAL': 'Rural'})
    df['Property_Area'] = df['Property_Area'].map({'Urban': 0, 'Semiurban': 1, 'Rural': 2})
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Total Income Log'] = np.log(df['Total_Income'])
    df['Loan_Amount_Log'] = np.log(df['LoanAmount'])
    df = df.dropna(subset=['Loan_Status'])
    print("\nCrosstab of Credit History vs Loan Status after preprocessing:")
    print(pd.crosstab(df['Credit_History'], df['Loan_Status'], margins=True))
    print('*****************************************************************************')
    # Divide dataset into independent and dependent variables
    X = df.iloc[:, np.r_[1:12, -2:]].values
    y = df.iloc[:, 12].values

    # Splitting the dataset into training and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Convert Categorical Variables into Numerical
    labelencoder_X = LabelEncoder()
    for i in range(0, 5):
        X_train[:, i] = labelencoder_X.fit_transform(X_train[:, i])
        X_test[:, i] = labelencoder_X.transform(X_test[:, i])

    labelencoder_y = LabelEncoder()
    y_train = labelencoder_y.fit_transform(y_train)
    y_test = labelencoder_y.transform(y_test)

    # Standardize the data values into standard format
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    return X_train, X_test, y_train, y_test
load_and_preprocess_data('Loan Data.csv')


# #### Model Evaluation

# In[3]:


def evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f'{model_name}:')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('F1 Score:', f1_score(y_test, y_pred))
    print(f'Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}')
    plot_confusion_matrix(y_test, y_pred, model_name)
    print('\n')

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Approved', 'Approved'], yticklabels=['Not Approved', 'Approved'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# #### Hyperparameter tuning and Evaluation Function

# In[4]:


def tune_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        'Naive Bayes': GaussianNB(),
        'SVC': SVC(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }
    
    params = {
        'Naive Bayes': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        },
        'SVC': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1],
            'max_depth': [3, 4, 5]
        },
        'Decision Tree': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        }
    }
    
    best_model = None
    best_score = 0
    best_model_name = None

    for model_name, model in models.items():
        print(f"Tuning {model_name}...")
        if model_name in params:
            random_search = RandomizedSearchCV(model, params[model_name], n_iter=10, cv=3, random_state=0, scoring='accuracy')
            random_search.fit(X_train, y_train)
            y_pred = random_search.best_estimator_.predict(X_test)
            print(f"Best Score for {model_name}: {accuracy_score(y_test, y_pred)}")
            print(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}")
            plot_confusion_matrix(y_test, y_pred, model_name)
            if accuracy_score(y_test, y_pred) > best_score:
                best_score = accuracy_score(y_test, y_pred)
                best_model = random_search.best_estimator_
                best_model_name = model_name
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(f"Score for {model_name}: {accuracy_score(y_test, y_pred)}")
            print(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}")
            plot_confusion_matrix(y_test, y_pred, model_name)
            if accuracy_score(y_test, y_pred) > best_score:
                best_score = accuracy_score(y_test, y_pred)
                best_model = model
                best_model_name = model_name

    print(f"The Best Model is: {best_model_name} with an accuracy of {best_score}")
    return best_model


# #### Train and Save Best model

# In[5]:


def train_and_save_best_model(X_train, y_train, X_test, y_test):
    # Tune models and get the best one
    best_model = tune_and_evaluate_models(X_train, y_train, X_test, y_test)
    
    # Save the best model to a file
    joblib.dump(best_model, 'Loan_Status_Predict.pkl')
    
    return best_model


# In[ ]:





# #### Graphical User Interface

# In[6]:


def predict_loan_status():
    # Retrieve user inputs
    gender = int(gender_entry.get())
    married = int(married_entry.get())
    dependents = int(dependents_entry.get())
    education = int(education_entry.get())
    self_employed = int(self_employed_entry.get())
    applicant_income = float(applicant_income_entry.get())
    coapplicant_income = float(coapplicant_income_entry.get())
    loan_amount = float(loan_amount_entry.get())
    loan_amount_term = float(loan_amount_term_entry.get())
    credit_history = float(credit_history_entry.get())
    property_area = int(property_area_entry.get())
    
    # Load the trained model
    model = joblib.load('Loan_Status_Predict.pkl')
    
    # Create a DataFrame with user inputs
    user_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })
    
    # Make prediction
    result = model.predict(user_data)
    
    # Show prediction result
    if result == 1:
        messagebox.showinfo("Prediction Result", "Loan Approved!")
    else:
        messagebox.showinfo("Prediction Result", "Loan Not Approved")

def create_gui():
    root = Tk()
    root.title("Loan Status Prediction")

    # Create input fields and labels
    Label(root, text="Gender [1:Male, 0:Female]:").grid(row=0, column=0)
    Label(root, text="Married [1:Yes, 0:No]:").grid(row=1, column=0)
    Label(root, text="Dependents:").grid(row=2, column=0)
    Label(root, text="Education [1:Graduate, 0:Not Graduate]:").grid(row=3, column=0)
    Label(root, text="Self Employed [1:Yes, 0:No]:").grid(row=4, column=0)
    Label(root, text="Applicant Income:").grid(row=5, column=0)
    Label(root, text="Coapplicant Income:").grid(row=6, column=0)
    Label(root, text="Loan Amount:").grid(row=7, column=0)
    Label(root, text="Loan Amount Term:").grid(row=8, column=0)
    Label(root, text="Credit History [1:Yes, 0:No]:").grid(row=9, column=0)
    Label(root, text="Property Area [0:Urban, 1:Semiurban, 2:Rural]:").grid(row=10, column=0)

    # Create entry fields
    global gender_entry, married_entry, dependents_entry, education_entry, self_employed_entry, applicant_income_entry, coapplicant_income_entry, loan_amount_entry, loan_amount_term_entry, credit_history_entry, property_area_entry

    gender_entry = Entry(root)
    married_entry = Entry(root)
    dependents_entry = Entry(root)
    education_entry = Entry(root)
    self_employed_entry = Entry(root)
    applicant_income_entry = Entry(root)
    coapplicant_income_entry = Entry(root)
    loan_amount_entry = Entry(root)
    loan_amount_term_entry = Entry(root)
    credit_history_entry = Entry(root)
    property_area_entry = Entry(root)

    # Position entry fields
    gender_entry.grid(row=0, column=1)
    married_entry.grid(row=1, column=1)
    dependents_entry.grid(row=2, column=1)
    education_entry.grid(row=3, column=1)
    self_employed_entry.grid(row=4, column=1)
    applicant_income_entry.grid(row=5, column=1)
    coapplicant_income_entry.grid(row=6, column=1)
    loan_amount_entry.grid(row=7, column=1)
    loan_amount_term_entry.grid(row=8, column=1)
    credit_history_entry.grid(row=9, column=1)
    property_area_entry.grid(row=10, column=1)

    # Create predict button
    predict_button = Button(root, text="Predict", command=predict_loan_status)
    predict_button.grid(row=11, columnspan=2)

    root.mainloop()


# #### Main Function

# In[7]:



def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data('Loan Data.csv')
    models = {
        'Naive Bayes': GaussianNB(),
        'SVC': SVC(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }

    for model_name, model in models.items():
        evaluate_model(model, model_name, X_train, y_train, X_test, y_test)
    
    
    # Train models and save the best one
    train_and_save_best_model(X_train, y_train, X_test, y_test)
    
    # Run the GUI
    create_gui()

# Execute main function
main()


# In[ ]:





# In[ ]:




