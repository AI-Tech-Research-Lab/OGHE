import json
import numpy as np
import os
from scipy.stats import chi2_contingency

def load_confusion_matrices(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    model_names = list(data.keys())
    if len(model_names) != 2:
        raise ValueError("JSON file must contain exactly two models")
    conf_matrix1 = np.array(data[model_names[0]])
    conf_matrix2 = np.array(data[model_names[1]])
    return conf_matrix1, conf_matrix2, model_names

def mcnemar_bowker_test(conf_matrix1, conf_matrix2):
    # Check if the two confusion matrices have the same shape
    assert conf_matrix1.shape == conf_matrix2.shape, "Confusion matrices must have the same shape"
    
    n = conf_matrix1.shape[0]
    
    # Calculate the difference matrix
    diff_matrix = conf_matrix1 - conf_matrix2
    
    # Convert the sum_matrix to float to avoid integer overflow
    sum_matrix = conf_matrix1 + conf_matrix2
    sum_matrix = sum_matrix.astype(float)
    sum_matrix[sum_matrix == 0] = np.inf  # Use np.inf to avoid division by zero
    
    # Calculate the statistic
    b_stat = np.sum((diff_matrix - diff_matrix.T)**2 / sum_matrix)
    
    # Ensure the observed matrix has non-negative values
    observed = np.abs(diff_matrix)
    p_value = chi2_contingency(observed, correction=False)[1]
    
    return b_stat, p_value

# Directory containing the JSON files
directory = 'confusion_matrices'

# Iterate through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        try:
            # Load confusion matrices from JSON file
            conf_matrix1, conf_matrix2, model_names = load_confusion_matrices(file_path)
            
            # Perform McNemar-Bowker test
            b_stat, p_value = mcnemar_bowker_test(conf_matrix1, conf_matrix2)
            
            # Print the results
            print(f"#####################################################")
            print(f"File: {filename}")
            print(f"McNemar-Bowker Statistic for {model_names[0]} vs {model_names[1]}: p-value: {p_value}")
            if p_value < 0.05:
                print("There is a statistically significant difference.")
            else:
                print("There is no statistically significant difference.")
        except ValueError as e:
            print(f"Error in file {filename}: {e}")
