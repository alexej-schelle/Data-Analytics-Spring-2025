import pandas as pd

def calculate_extreme_values(file_path, feature_name):
    """
    Calculate the extreme values (minimum and maximum) of a dataset for a specific feature.

    Parameters:
        file_path (str): Path to the CSV file containing the dataset.
        feature_name (str): The name of the feature/column to analyze.

    Returns:
        dict: A dictionary containing the minimum and maximum values of the feature.
    """
    # Load the dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"Error loading file: {e}"
    
    # Check if the feature exists in the dataset
    if feature_name not in df.columns:
        return f"Feature '{feature_name}' not found in the dataset."
    
    # Calculate extreme values
    min_value = df[feature_name].min()
    max_value = df[feature_name].max()
    
    return {
        "Minimum Value": min_value,
        "Maximum Value": max_value
    }

print(calculate_extreme_values('mba_decision_dataset.csv', 'Person ID'))
print(calculate_extreme_values('mba_decision_dataset.csv', 'Age'))
print(calculate_extreme_values('mba_decision_dataset.csv', 'Undergraduate GPA'))
print(calculate_extreme_values('mba_decision_dataset.csv', 'Annual Salary (Before MBA)'))
print(calculate_extreme_values('mba_decision_dataset.csv', 'GRE/GMAT Score'))
print(calculate_extreme_values('mba_decision_dataset.csv', 'Undergrad University Ranking'))
print(calculate_extreme_values('mba_decision_dataset.csv', 'MBA Funding Score'))
print(calculate_extreme_values('mba_decision_dataset.csv', 'Expected Post-MBA Salary'))

