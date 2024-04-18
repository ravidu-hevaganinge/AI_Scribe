import pandas as pd
import random

# Possible values for each parameter
names = ["John", "Jane", "Alice", "Bob", "Diane", "Gary"]
ages = ["25", "30", "35", "40", "45", "50"]
genders = ["male", "female"]
medical_histories = ["diabetes and hypertension", "hypertension and asthma", "depression and anxiety"]
visit_reasons = ["routine check-up", "emergency room follow-up", "annual physical"]
locations = ["the emergency room", "the clinic", "the office"]
patient_experiences = [
    "I felt dizzy and passed out.", 
    "I had a severe headache and felt nauseous.", 
    "I was experiencing chest pains and shortness of breath."
]
symptoms = ["Yes, I had a severe headache.", "No, I did not have any headache."]
blood_pressure_histories = ["has been normal", "has been high"]
traveling_responses = ["Yes, I travel frequently.", "No, I rarely travel."]
cuff_responses = ["Yes, I have my own blood pressure cuff.", "No, I don't have one."]
medications = ["lisinopril", "metoprolol", "amlodipine"]
medication_adherences = ["Yes, regularly", "No, sometimes I forget"]
other_conditions = ["depression", "asthma", "diabetes"]
condition_managements = [
    "I manage it with medication and regular visits to the doctor.", 
    "I am not doing well with managing my condition."
]
previous_times = ["last week", "yesterday", "this morning"]
additional_symptoms = ["No other symptoms.", "I've been coughing a lot too.", "I feel very tired recently."]

# Number of records
n = 100

# Randomly generate data for each parameter
data = {
    "name": [random.choice(names) for _ in range(n)],
    "age": [random.choice(ages) for _ in range(n)],
    "gender": [random.choice(genders) for _ in range(n)],
    "medical_history": [random.choice(medical_histories) for _ in range(n)],
    "visit_reason": [random.choice(visit_reasons) for _ in range(n)],
    "location": [random.choice(locations) for _ in range(n)],
    "patient_experience": [random.choice(patient_experiences) for _ in range(n)],
    "symptoms": [random.choice(symptoms) for _ in range(n)],
    "blood_pressure_history": [random.choice(blood_pressure_histories) for _ in range(n)],
    "traveling_response": [random.choice(traveling_responses) for _ in range(n)],
    "cuff_response": [random.choice(cuff_responses) for _ in range(n)],
    "medication": [random.choice(medications) for _ in range(n)],
    "medication_adherence": [random.choice(medication_adherences) for _ in range(n)],
    "other_condition": [random.choice(other_conditions) for _ in range(n)],
    "condition_management": [random.choice(condition_managements) for _ in range(n)],
    "previous_time": [random.choice(previous_times) for _ in range(n)],
    "additional_symptoms": [random.choice(additional_symptoms) for _ in range(n)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('C:\\vitAI\\cleaned_data\\synthetic_patients\\patient_parameters.csv', index=False)
print("CSV file 'patient_parameters.csv' created with randomized patient parameters.")
