import os
import sys

import pandas as pd
import numpy
import math

import matplotlib.pyplot as plt

# Load the data in 'DataDictionary.csv' with the path to your CSV file
df = pd.read_csv('mba_decision_dataset.csv')

print(' ')
print(df.info())
print(' ')

print(df.head())
print(df.count())

# Load the data in 'your_file.csv' with the path to your CSV file
df = pd.read_csv('SurveyData.csv')

print(' ')
print(df.info())
print(' ')

print(df.head())
print(df.count())

# Load the data in 'your_file.csv' with the path to your CSV file
df = pd.read_csv('TravelData.csv')

print(' ')
print(df.info())
print(' ')

print(df.head())
print(df.count())

# Unterscheidung als Numerische und Logische VariablenTypen
reduced_numeric = df[['Age', 'Travel_Distance', 'DepartureDelay_in_Mins', 'ArrivalDelay_in_Mins']].copy()
reduced_logic = df[['ID', 'Gender', 'CustomerType', 'TypeTravel', 'Travel_Class']].copy()

# Unterscheidung als Gender-spezifische VariablenTypen
reduced_female = df[df['Gender'] == 'Female'].copy()
reduced_male = df[df['Gender'] == 'Male'].copy()

print(reduced_numeric.mean())

print(reduced_female.head())
print(reduced_male.head())

# Unterscheidung als Gender-spezifische und numerische VariablenTypen
reduced_female_numeric = reduced_female[['Age', 'Travel_Distance', 'DepartureDelay_in_Mins', 'ArrivalDelay_in_Mins']].copy()
reduced_male_numeric = reduced_male[['Age', 'Travel_Distance', 'DepartureDelay_in_Mins', 'ArrivalDelay_in_Mins']].copy()

# Unterscheidungx als Gender-spezifische und logische VariablenTypen
reduced_female_logic = reduced_female[['ID', 'Gender', 'CustomerType', 'TypeTravel', 'Travel_Class']].copy()
reduced_male_logic = reduced_male[['ID', 'Gender', 'CustomerType', 'TypeTravel', 'Travel_Class']].copy()

print(reduced_female_numeric.head())
print(reduced_male_numeric.head())

print(reduced_female_logic.head())
print(reduced_male_logic.head())

# Create histograms
df.hist(bins=100, figsize=(8, 6))

# Show the plots
plt.show()

