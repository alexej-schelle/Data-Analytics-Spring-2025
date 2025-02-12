import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# Load the data
df = pd.read_csv('mba_decision_dataset.csv')

# Data Preprocessing
df_reduced = df.replace({'Male':1, 'Female':2, 'Economics':3, 'Science':4, 'Arts':5, 'Engineering':6, 'Business':7, 'Entrepreneur':8, 'Entrepreneurship':8.5 , 'Analyst':9, 'Engineer':10, 'Consultant':11, 'Manager':12, 'Yes':13, 'No':14, 'Employer':15, 'Loan':16, 'Scholarship':17, 'Self-funded':18, 'Executive':19, 'Marketing Director':20, 'Consutant':21, 'Startup Founder':22, 'Finance Manager':23, 'International':24, 'Domestic':25, 'Networking':26, 'Career Growth':27, 'Skill Enhancement':28, 'Entrepreneuership':29, 'On-Campus':30, 'Online':31, 'Yes':32, 'No':33, 'Other': 34})

# Step 2: Evaluate distributions

# Create histograms
df_reduced.hist(bins=50, figsize=(8, 6))

# Add title and labels
plt.title('Histogram Example')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()

# Define random gauss variable around mean with standard deviation

mean = df_reduced.mean()
variance = df_reduced.var()

random_var = random.gauss(mean, variance)

print(random_var)

# Define random uniform variable with minimum and maximum

minimum = df_reduced.min()
maximum = df_reduced.max()

random_var = random.uniform(minimum, maximum)
print(random_var)