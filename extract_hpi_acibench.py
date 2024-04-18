import pandas as pd
import re

# Function to extract relevant sections from the full clinical note
def extract_relevant_sections(full_note):
    pattern = r"CHIEF COMPLAINT\r\n\r\n(.*?)\r\n\r\nHISTORY OF PRESENT ILLNESS\r\n\r\n(.*?)(?:\r\n\r\n|$)"
    match = re.search(pattern, full_note, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    else:
        return "",""

# Load your dataset here
# This should be a DataFrame loaded from your dataset file which includes at least two columns: 'dialogue' and 'full_note'
data = pd.read_csv("C:\\vitAI\\aci-bench\\data\\src_experiment_data\\test1_aci_asrcorr.csv")

# Applying the function to each row in the DataFrame
data[['chief_complaint', 'history_present_illness']] = data.apply(lambda row: pd.Series(extract_relevant_sections(row['note'])), axis=1)

# Creating a new DataFrame with the desired structure
result_df = pd.DataFrame({
    'dataset': 'ACI-BENCH',
    'id': data.index,
    'dialogue': data['dialogue'],
    'note': data['chief_complaint'] + "\n\n" + data['history_present_illness']
})

print(result_df["note"])

# Some of the dialogues don't contain a history of present illness section!
print(f"example note which did not work with regular expression: {data["note"][5]}")

# Save to CSV
result_df.to_csv('filtered_clinical_notes.csv', index=False)
