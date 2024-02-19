"""
< ALY 6140 GROUP 3 , Capstone project lib >

Written by Bo Liu, Yuning Chen

# This is a Python script that contains 3 functions to support Group 3's Capstone Project 
Jupyter Notebook presentation

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.stats import weightstats as weightstats

def deposit_sum_by_column(bank, column_name):
    """
    This function calculates the deposit ratio for each category in the specified column 
    and visualizes the results using a bar plot.
    
    Parameters:
    bank (pd.DataFrame): The bank marketing dataset in a pandas DataFrame.
    column_name (str): The name of the column for which the deposit ratio should be calculated.
    """
  
    # Filter out only rows with 'yes' deposits and group by the specified column
    deposit_sum_by_column = bank[bank['deposit'] == 'yes'].groupby([column_name]).size().reset_index(name='count')

    # Calculate the total number of instances for each category in the specified column
    total_by_column = bank.groupby([column_name]).size().reset_index(name='total')

    # Merge the two DataFrames on the specified column
    deposit_sum_by_column = deposit_sum_by_column.merge(total_by_column, on=column_name)

    # Calculate the deposit ratio and add it as a new column
    deposit_sum_by_column['deposit_ratio'] = deposit_sum_by_column['count'] / deposit_sum_by_column['total']

    # Print the resulting DataFrame
    print(deposit_sum_by_column)

    # Visualize the results using a bar plot
    sns.set_theme(style="whitegrid")
    sns.barplot(x=column_name, y="deposit_ratio", data=deposit_sum_by_column)
    plt.xlabel(f"{column_name} category")
    plt.ylabel("Deposit ratio")
    plt.title(f"Deposit ratio by {column_name}")
    plt.xticks(rotation=45)  # Rotate x-axis labels
    plt.show()
    
    return

def add_factor_to_deposit_sum_by_job(bank, column_name):
    """
    This function calculates the deposit ratio for each combination of job and the specified 
    column categories, and visualizes the results using a grouped bar plot.
    
    Parameters:
    bank (pd.DataFrame): The bank marketing dataset in a pandas DataFrame.
    column_name (str): The name of the column for which the deposit ratio should be calculated.
    """

    # Calculate the number of 'yes' deposits for each combination of job and specified column categories
    deposit_sum_by_job = bank[bank['deposit'] == 'yes'].groupby(['job', column_name]).size().reset_index(name='count')

    # Calculate the total number of instances for each combination of job and specified column categories
    total_by_job = bank.groupby(['job', column_name]).size().reset_index(name='total')

    # Merge the two DataFrames on the 'job' and specified column
    deposit_sum_by_job = deposit_sum_by_job.merge(total_by_job, on=['job', column_name])

    # Calculate the deposit ratio and add it as a new column
    deposit_sum_by_job['deposit_ratio'] = deposit_sum_by_job['count'] / deposit_sum_by_job['total']

    # Print the resulting DataFrame
    print(deposit_sum_by_job)

    # Visualize the results using a grouped bar plot
    sns.set_theme(style="whitegrid")
    sns.barplot(x="job", y="deposit_ratio", hue=column_name, data=deposit_sum_by_job)
    plt.xlabel("Job category")
    plt.ylabel("Deposit ratio")
    plt.title(f"Deposit ratio by job and {column_name} status")
    plt.xticks(rotation=45)  # Rotate x-axis labels
    plt.show()
    
    return

def t_weightstates_on_feature(bank, column_name):
    """
    This function calculates and prints the mean values of the specified column for 
    deposits and no-deposits, and performs an independent t-test between them.
    
    Parameters:
    bank (pd.DataFrame): The bank marketing dataset in a pandas DataFrame.
    column_name (str): The name of the column for which the t-test should be performed.
    """
    
    # Filter out the specified column values for 'yes' and 'no' deposits
    deposit_balance = bank.loc[bank['deposit']=='yes', column_name]
    no_deposit_balance = bank.loc[bank['deposit']=='no', column_name]   
    
    # Print the mean values
    print(f"The mean {column_name} of deposit is:    {np.mean(deposit_balance)}")
    print(f"The mean {column_name} of no_deposit is: {np.mean(no_deposit_balance)}")
    
    # Perform an independent t-test
    t_stat, p_val, dof = weightstats.ttest_ind(deposit_balance, no_deposit_balance)
    
    # Print the results of the t-test
    print(f"The t-statistic is: {t_stat}")
    print(f"The p-value is: {p_val}")
    print(f"The degrees of freedom is: {dof}")
    
    return

def evaluate_model(model, X_test, Y_test):
    """
    This function evaluates the performance of a given model on the test data.
    
    Parameters:
    model (Model object): The machine learning model to be evaluated.
    X_test (DataFrame or ndarray): The test input data.
    Y_test (DataFrame or ndarray): The test target data.
    
    Returns:
    dict: A dictionary containing various performance metrics.
    """
    
    from sklearn import metrics

    # Predict Test Data 
    Y_prediction = model.predict(X_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(Y_test, Y_prediction)  # Accuracy
    prec = metrics.precision_score(Y_test, Y_prediction)  # Precision
    rec = metrics.recall_score(Y_test, Y_prediction)  # Recall
    f1 = metrics.f1_score(Y_test, Y_prediction)  # F1-score
    kappa = metrics.cohen_kappa_score(Y_test, Y_prediction)  # Kappa score

    # Calculate area under curve (AUC)
    Y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(Y_test, Y_pred_proba)  # ROC curve components
    auc = metrics.roc_auc_score(Y_test, Y_pred_proba)  # AUC

    # Display confusion matrix
    cm = metrics.confusion_matrix(Y_test, Y_prediction)  # Confusion matrix

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa, 
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}

