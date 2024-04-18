import os
import pandas as pd
import re

def extract_relevant_sections(full_note):
    # pattern = r"CHIEF COMPLAINT\r\n\r\n(.*?)\r\n\r\nHISTORY OF PRESENT ILLNESS\r\n\r\n(.*?)(?:\r\n\r\n|$)"
    pattern2 = r"CHIEF COMPLAINT\r\n\r\n(.*?)\r\n\r\nHISTORY OF PRESENT ILLNESS\r\n\r\n(.*?)(?=\r\n\r\n[A-Z])"
    pattern = r"CHIEF COMPLAINT\r\n\r\n(.*?)\r\n\r\nHISTORY OF PRESENT ILLNESS\r\n\r\n(.*?)(?=\r\n\r\n[A-Z ]+\r\n)"
    match = re.search(pattern, full_note, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    else:
        return "",""

# Define paths to the subdirectories
base_path = 'C:\\vitAI\\aci-bench\\data'
sub_dirs = ['challenge_data', 'src_experiment_data']

# Prepare dataframes to hold the extracted data
columns = ['dataset', 'id', 'dialogue', 'note']
train_df = pd.DataFrame(columns=columns)
test_df = pd.DataFrame(columns=columns)
valid_df = pd.DataFrame(columns=columns)

# Process files in each subdirectory
for sub_dir in sub_dirs:
    full_path = os.path.join(base_path, sub_dir)
    for file_name in os.listdir(full_path):
        if file_name.endswith(".csv") and not file_name.endswith('metadata.csv'):  # Only process CSV files
            print(file_name)
            file_path = os.path.join(full_path, file_name)
            data = pd.read_csv(file_path)
            
            # Process each row in the dataframe
            for index, row in data.iterrows():
                chief_complaint, hpi = extract_relevant_sections(row['note'])
                print(hpi)
                processed_data = pd.DataFrame({
                    'dataset': sub_dir,
                    'id': data.index,
                    'dialogue': row['dialogue'],
                    'note': f"CHIEF COMPLAINT\n\n{chief_complaint}\n\nHISTORY OF PRESENT ILLNESS\n\n{hpi}"
                })
                
                # Append to appropriate dataframe based on the file designation
                if 'train' in file_name:
                    # train_df = train_df.append(processed_data, ignore_index=True)
                    train_df = pd.concat([train_df, processed_data], ignore_index=True)
                elif 'test' in file_name:
                    # test_df = test_df.append(processed_data, ignore_index=True)
                    test_df = pd.concat([test_df, processed_data], ignore_index=True)
                elif 'valid' in file_name:
                    # valid_df = valid_df.append(processed_data, ignore_index=True)
                    valid_df = pd.concat([valid_df, processed_data], ignore_index=True)
                break

# Save the dataframes to CSV files
train_df.to_csv('C:\\vitAI\\cleaned_data\\cleaned_aci-bench\\train.csv', index=False)
test_df.to_csv('C:\\vitAI\\cleaned_data\\cleaned_aci-bench\\test.csv', index=False)
valid_df.to_csv('C:\\vitAI\\cleaned_data\\cleaned_aci-bench\\valid.csv', index=False)

print("Data processing complete. Files saved: train.csv, test.csv, valid.csv")
