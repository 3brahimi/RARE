import pandas as pd
import json
import math

# Load the CSV file
df = pd.read_csv('./FeynmanEquations.csv')

# Iterate over the rows of the DataFrame
for _, row in df.iterrows():
    # Check if the number of variables is NaN
    if pd.isnull(row['# variables']):
        continue

    # Initialize the output dictionary for each equation
    output = {
        "equation": row['Formula'],
        "models": [
            {
                "type": "linear",
                "model_path": "clean",
                "load": False,
                "training_type": "clean",
                "fit_args": {
                    "epochs": 200,
                    "early_stopping": 20
                },
                "noisy_input_feats": [],
                "extended_data": True,
                "dp": False
            }            
        ],
        "features": {}
    }

    # For each variable in the row
    for i in range(1, int(row['# variables']) + 1):
        # Get the variable name, low, and high values
        var_name = row[f'v{i}_name']
        var_low = row[f'v{i}_low']
        var_high = row[f'v{i}_high']

        # Check if the values are NaN
        if pd.isnull(var_name) or math.isnan(var_low) or math.isnan(var_high):
            continue

        # Add the variable to the features dictionary
        output['features'][var_name] = {
            "type": "float",
            "range": [var_low, var_high]
        }
        # Add the variable index to the noisy_input_feats list
        output['models'][0]['noisy_input_feats'].append(i - 1)
    # Convert the dictionary to a JSON string
    output_json = json.dumps(output, indent=4)
    print(output_json)
    # Save the JSON string to a file
    filename = row['Filename'].replace(' ', '_')
    filename = filename.replace('.', '_')+ '.json'
    with open(f"../configs/equations/{filename}", 'w') as f:
        f.write(output_json)
