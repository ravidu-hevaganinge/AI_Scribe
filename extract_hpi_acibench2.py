import pandas as pd
import os
import re
import fnmatch

# Define the base path where the subdirectories are located
base_path = 'C:\\vitAI\\aci-bench\\data'

# List the subdirectories to search in
subdirectories = ['challenge_data', 'src_experiment_data']

# Prepare a list to collect DataFrames
dataframes = []

# Regular expression to extract the required sections
pattern = re.compile(r"CHIEF COMPLAINT\r\n\r\n(.*?)\r\n\r\nHISTORY OF PRESENT ILLNESS\r\n\r\n(.*?)(?:\r\n\r\n|$)", re.DOTALL)

# Function to extract chief complaint and history of present illness
# def extract_note_sections(full_note):
#     match = pattern.search(full_note)
#     if match:
#         return match.group(1).strip(), match.group(2).strip()
#     return "", ""

def extract_relevant_sections(full_note):
    pattern = r"CHIEF COMPLAINT\r\n\r\n(.*?)\r\n\r\nHISTORY OF PRESENT ILLNESS\r\n\r\n(.*?)(?:\r\n\r\n|$)"
    match = re.search(pattern, full_note, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    else:
        return "",""
    

training_filenames = []
testing_filenames = []
validation_filenames = []
# Process each subdirectory
for subdirectory in subdirectories:
    dir_path = os.path.join(base_path, subdirectory)
    for filename in os.listdir(dir_path):
        if filename.endswith('.csv') and not filename.endswith('metadata.csv'):
            if fnmatch.fnmatch(filename, '*valid_'):
                continue
            print(filename)
            # Construct file path
            file_path = os.path.join(dir_path, filename)
            # Read the dataset
            df = pd.read_csv(file_path)
            
            # Apply the function to extract sections
            # df['chief_complaint'], df['history_of_present_illness'] = zip(*df['note'].map(extract_note_sections))
            df[['chief_complaint', 'history_of_present_illness']] = df.apply(lambda row: pd.Series(extract_relevant_sections(row['note'])), axis=1)
            # df['chief_complaint'], df['history_of_present_illness'] = zip(*df['note'].map(extract_relevant_sections))
            
            # Create a new DataFrame with the necessary columns
            new_df = pd.DataFrame({
                'dataset': 'ACI-BENCH',
                'id': df.index,
                'dialogue': df['dialogue'],
                'note': df['chief_complaint'] + "\n\n" + df['history_of_present_illness']
            })
            
            # Append the new DataFrame to the list
            dataframes.append(new_df)

# Concatenate all DataFrames in the list into a single DataFrame
final_df = pd.concat(dataframes, ignore_index=True)

print(final_df)

# Save the final DataFrame to a new CSV file
# final_df.to_csv('C:\\vitAI\\cleaned_data\\cleaned_aci-bench\\extracted_notes.csv', index=False)

# print("Data extraction complete. Saved to extracted_notes.csv")
