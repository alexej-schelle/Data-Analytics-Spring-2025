import os
import sys

import pandas as pd
import numpy
import math

import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('mba_decision_dataset.csv')

# Display the first few rows of the DataFrame
print(df.head())

# Show number of classes (elements per class)
print(df.count())

# Check for datatype of classes
print(df.dtypes)

# Calculate mean value and standard deviations within each numeric class 

# Project on numeric classes

# Unterscheidung als Numerische und Logische VariablenTypen
reduced_numeric = df[['Person ID', 'Age', 'Undergraduate GPA', 'Years of Work Experience', 'Annual Salary (Before MBA)', 'GRE/GMAT Score', 'Expected Post-MBA Salary']].copy()
reduced_logic = df[['Gender', 'Undergraduate Major', 'Current Job Title', 'Has Management Experience', 'Undergrad University Ranking', 'Entrepreneurial Interest', 'Networking Importance', 'MBA Funding Source', 'Desired Post-MBA Role' ,'Location Preference (Post-MBA)' , 'Reason for MBA' , 'Online vs. On-Campus MBA', 'Decided to Pursue MBA?']].copy()

print(' ')
print('Mean Values : ')
print(' ')
print(reduced_numeric.mean())
print(' ')
print('Standard Deviations : ')
print(' ')
print(reduced_numeric.std())

print(' ')
print('Counting Statistcs: ')
print(' ')
print(reduced_logic['Gender'].value_counts())
print(' ')
print(' ')
print('Counting Statistcs: ')
print(' ')
print(reduced_logic['Undergraduate Major'].value_counts())
print(' ')
print(' ')
print('Counting Statistcs: ')
print(' ')
print(reduced_logic['Current Job Title'].value_counts())
print(' ')
print(' ')
print('Counting Statistcs: ')
print(' ')
print(reduced_logic['Has Management Experience'].value_counts())
print(' ')
print(' ')
print('Counting Statistcs: ')
print(' ')
print(reduced_logic['Undergrad University Ranking'].value_counts())
print(' ')
print(' ')
print('Counting Statistcs: ')
print(' ')
print(reduced_logic['Entrepreneurial Interest'].value_counts())
print(' ')
print(' ')
print('Counting Statistcs: ')
print(' ')
print(reduced_logic['Networking Importance'].value_counts())
print(' ')
print(' ')
print('Counting Statistcs: ')
print(' ')
print(reduced_logic['MBA Funding Source'].value_counts())
print(' ')
print(' ')
print('Counting Statistcs: ')
print(' ')
print(reduced_logic['Desired Post-MBA Role'].value_counts())
print(' ')
print(' ')
print('Counting Statistcs: ')
print(' ')
print(reduced_logic['Location Preference (Post-MBA)'].value_counts())
print(' ')
print(' ')
print('Counting Statistcs: ')
print(' ')
print(reduced_logic['Reason for MBA'].value_counts())
print(' ')
print(' ')
print('Counting Statistcs: ')
print(' ')
print(reduced_logic['Online vs. On-Campus MBA'].value_counts())
print(' ')
print(' ')
print('Counting Statistcs: ')
print(' ')
print(reduced_logic['Decided to Pursue MBA?'].value_counts())
print(' ')
print(' ')

# Calcluate correlations of numeric variables

# Standard (quadratic) correlations:
print(' ')
print('Quadratic correlations: ')
print(' ')
print(reduced_numeric.corr())
print(' ')

# Standard (quadratic) correlations:
print(' ')
print('Pearson correlations: ')
print(' ')
print(reduced_numeric.corr('pearson'))
print(' ')

# Standard (quadratic) correlations:
print(' ')
print('Spearman correlations: ')
print(' ')
print(reduced_numeric.corr('spearman'))
print(' ')

# Calculate correlations between logic variables:
df_reduced_logic_mapped = reduced_logic.replace({'Has Management Experience': 1, 'GRE/GMAT Score': 2, 'Undergrad University Ranking': 3, 'Entrepreneurial Interest': 4, 'Networking Importance': 5, 'MBA Funding Source': 6, 'Desired Post-MBA Role': 7, 'Expected Post-MBA Salary': 8, 'Location Preference (Post-MBA)': 9, 'Reason for MBA' : 10, 'Online vs. On-Campus MBA' : 11, 'Decided to Pursue MBA?' : 12, 'Male': 13, 'Female' : 14, 'Other': 15, 'Economics': 16, 'Science' : 17, 'Arts' : 18, 'Engineering' : 19, 'Business' : 20, 'Entrepreneur': 21, 'Analyst': 22, 'Engineer': 23, 'Consultant': 24, 'Manager': 25,'No':26, 'Yes': 27, 'Loan': 28, 'Scholarship': 29, 'Self-funded': 30, 'Employer': 31, 'Finance Manager': 32, 'Startup Founder': 33, 'Marketing Director': 34, 'Executive': 35, 'International': 36, 'Domestic': 37, 'Entrepreneurship':38, 'Career Growth': 39, 'Skill Enhancement': 40, 'Networking' : 41, 'On-Campus': 42, 'Online': 43})

print(df_reduced_logic_mapped.corr())
print(df_reduced_logic_mapped.corr('pearson'))
print(df_reduced_logic_mapped.corr('spearman'))
