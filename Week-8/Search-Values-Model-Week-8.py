import pandas as pd

# Specify the path to your CSV file
file_path = "mba_decision_dataset.csv"  # Replace with your actual file path

# Load the data into a pandas DataFrame
try:
    df = pd.read_csv(file_path)
    print("Data successfully loaded!")
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

# Step 1: Identify non-numeric columns
non_numeric_columns = df.select_dtypes(exclude=["number"]).columns

# Step 2: Summarize non-numeric columns
for col in non_numeric_columns:
    print(f"Summary for column: {col}")
    print(df[col].value_counts())  # Count unique non-numeric values
    print("-" * 50)

df_reduced = df[['Gender', 'Undergraduate Major', 'Undergraduate GPA', 'Current Job Title', 'Has Management Experience', 'GRE/GMAT Score', 'Undergrad University Ranking', 'Entrepreneurial Interest', 'Networking Importance', 'MBA Funding Source', 'Desired Post-MBA Role', 'Location Preference (Post-MBA)', 'Reason for MBA', 'Online vs. On-Campus MBA', 'Decided to Pursue MBA?']]
df_reduced = df_reduced.replace({'Male':1, 'Female':2, 'Economics':3, 'Science':4, 'Arts':5, 'Engineering':6, 'Business':7, 'Entrepreneur':8, 'Entrepreneurship':8.5 , 'Analyst':9, 'Engineer':10, 'Consultant':11, 'Manager':12, 'Yes':13, 'No':14, 'Employer':15, 'Loan':16, 'Scholarship':17, 'Self-funded':18, 'Executive':19, 'Marketing Director':20, 'Consutant':21, 'Startup Founder':22, 'Finance Manager':23, 'International':24, 'Domestic':25, 'Networking':26, 'Career Growth':27, 'Skill Enhancement':28, 'Entrepreneuership':29, 'On-Campus':30, 'Online':31, 'Yes':32, 'No':33, 'Other': 34})

with open("output.txt", "w") as filename:

    print(' ')
    print(' ')
    print(df_reduced, file=filename)
    print(' ')
    print(' ')

print(df_reduced.mean())
print(df_reduced.std())
print(df_reduced.corr())

