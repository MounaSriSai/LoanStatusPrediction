# Loan Eligibility Prediction

## Overview

The Loan Eligibility Prediction project aims to develop a predictive model that determines whether a loan applicant is eligible for a loan based on various financial and personal features. This project utilizes machine learning techniques to analyze historical data and predict the likelihood of loan approval.

## Table of Contents

1. [Project Description](#project-description)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Data](#data)
6. [Model](#model)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

## Project Description

The goal of this project is to create a model that can predict loan eligibility based on several input features. The project involves data preprocessing, feature selection, model training, and evaluation. The final model helps in making data-driven decisions for loan approvals.

## Features

- **Data Preprocessing:** Cleaning and transforming raw data into a suitable format for modeling.
- **Feature Engineering:** Selecting and creating relevant features for improved model performance.
- **Model Training:** Implementing and training various machine learning algorithms.
- **Model Evaluation:** Assessing model performance using appropriate metrics.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/MounaSriSai/loan-eligibility-prediction.git
    ```

2. Navigate to the project directory:
    ```bash
    cd loan-eligibility-prediction
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your dataset in the `data/` directory.
2. Modify the configuration settings in `config.yaml` if necessary.
3. Run the main script to train and evaluate the model:
    ```bash
    python main.py
    ```

## Data

The dataset used in this project contains information about loan applicants and their eligibility status. It includes features such as:

- Applicant's age
- Applicant's income
- Loan amount
- Loan term
- Credit score
- Employment status

Ensure that the data is in the correct format and located in the `data/` directory.

## Model

The project explores various machine learning algorithms to predict loan eligibility, including:

- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- Naive Bayes
- Support Vector Machine


The best-performing model is selected based on evaluation metrics such as accuracy, precision, recall, and F1 score.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
