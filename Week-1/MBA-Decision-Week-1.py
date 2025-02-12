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

# Show number of categories with each class 
# First class 1
df_class = df['Age']
print(df_class.value_counts()) 

# First class 2
df_class = df['Gender']
print(df_class.value_counts()) 

# First class 3
df_class = df['Undergraduate Major']
print(df_class.value_counts()) 

# First class 4
df_class = df['Undergraduate GPA']
print(df_class.value_counts()) 

# First class 8
df_class = df['Years of Work Experience']
print(df_class.value_counts()) 

# First class 9
df_class = df['Current Job Title']
print(df_class.value_counts()) 

# First class 10
df_class = df['Annual Salary (Before MBA)']
print(df_class.value_counts()) 

# First class 11
df_class = df['Has Management Experience']
print(df_class.value_counts()) 

# First class 12
df_class = df['GRE/GMAT Score']
print(df_class.value_counts()) 

# First class 13
df_class = df['Undergrad University Ranking']
print(df_class.value_counts()) 

# First class 14
df_class = df['Entrepreneurial Interest']
print(df_class.value_counts()) 

# First class 15
df_class = df['Networking Importance']
print(df_class.value_counts()) 

# First class 16
df_class = df['MBA Funding Source']
print(df_class.value_counts()) 

# First class 17
df_class = df['Desired Post-MBA Role']
print(df_class.value_counts()) 

# First class 18
df_class = df['Expected Post-MBA Salary']
print(df_class.value_counts()) 

# First class 19
df_class = df['Location Preference (Post-MBA)']
print(df_class.value_counts()) 

# First class 20
df_class = df['Reason for MBA']
print(df_class.value_counts()) 

# First class 21
df_class = df['Online vs. On-Campus MBA']
print(df_class.value_counts()) 

# First class 22
df_class = df['Decided to Pursue MBA?']
print(df_class.value_counts()) 

# Calculate mean values for numeric types
print('Age: ', df['Age'].mean())

# Calculate mean values for numeric types
print('Undergraduate GPA: ', df['Undergraduate GPA'].mean())

# Calculate mean values for numeric types
print('Years of Work Experience: ', df['Years of Work Experience'].mean())

# Calculate mean values for numeric types
print('Annual Salary (Before MBA): ', df['Annual Salary (Before MBA)'].mean())

# Create histograms
df.hist(bins=50, figsize=(8, 6))

# Show the plots
plt.show()
