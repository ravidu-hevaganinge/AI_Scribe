import openai
import os
import pandas as pd

# Load the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_dialogue(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",  # specify the appropriate model
        prompt=prompt,
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].text.strip()

n = 100
new_dialogues = []

# Define the conversation template and extractable sections
conversation_template_virtscribe = """
[doctor] Hi {name}, how are you?
[patient] I'm doing okay, how are you?
[doctor] I'm doing okay. So I know the nurse told you about Dax and I'd like to tell Dax a little bit about you, okay?
[patient] Okay.
[doctor] {name} is a {age} year old {gender} with a past medical history significant for {medical_history} who presents for {visit_reason}.
[doctor] So {name}, what's going on? I heard that your blood pressure was really high at {location}. What happened?
[patient] {patient_experience}
[doctor] Did you have a headache?
[patient] {symptoms}
[doctor] Okay, all right. Have your blood pressures been running high in the past?
[patient] {blood_pressure_history}
[doctor] You're not taking your blood pressures, I take it, when you're traveling?
[patient] {traveling_response}
[doctor] Okay, but you did buy the cuff like we talked about in the past?
[patient] {cuff_response}
[doctor] And are you taking your medication, are you taking the {medication}?
[patient] {medication_adherence}
[doctor] Then in terms of your {other_condition}, how are you doing?
[patient] {condition_management}
[doctor] Okay, well great. So I know you did a review of system sheet when you checked in and I know that you had the headache and the lightheadedness at {previous_time}. Do you have any other symptoms at this time, chest pain, shortness of breath, anything like that?
[patient] {additional_symptoms}
[doctor] Okay, all right. Well I want to go ahead and do a quick physical exam, okay?
[patient] Okay.
"""

conversation_template_asrcorr = """
[doctor] Hi {name}, how are you?
[patient] I'm doing okay, how are you?
[doctor] I'm doing okay. So I know the nurse told you about Dax and I'd like to tell Dax a little bit about you, okay?
[patient] Okay.
[doctor] {name} is a {age} year old {gender} with a past medical history significant for {medical_history} who presents for {visit_reason}.
[doctor] So {name}, what's going on? I heard that your blood pressure was really high at {location}. What happened?
[patient] {patient_experience}
[doctor] Did you have a headache?
[patient] {symptoms}
[doctor] Okay, all right. Have your blood pressures been running high in the past?
[patient] {blood_pressure_history}
[doctor] You're not taking your blood pressures, I take it, when you're traveling?
[patient] {traveling_response}
[doctor] Okay, but you did buy the cuff like we talked about in the past?
[patient] {cuff_response}
[doctor] And are you taking your medication, are you taking the {medication}?
[patient] {medication_adherence}
[doctor] Then in terms of your {other_condition}, how are you doing?
[patient] {condition_management}
[doctor] Okay, well great. So I know you did a review of system sheet when you checked in and I know that you had the headache and the lightheadedness at {previous_time}. Do you have any other symptoms at this time, chest pain, shortness of breath, anything like that?
[patient] {additional_symptoms}
[doctor] Okay, all right. Well I want to go ahead and do a quick physical exam, okay?
[patient] Okay.
"""
# Define parameters for dialogue generation
parameters = {
    "name": "Diane",
    "age": "28",
    "gender": "female",
    "medical_history": "depression and hypertension",
    "visit_reason": "emergency room follow-up",
    "location": "the emergency room",
    "patient_experience": "I ended up going for a walk, it was sunny and great. I felt really light-headed and started to fall, but my boyfriend caught me.",
    "symptoms": "Yes, I did.",
    "blood_pressure_history": "Yes, they have been. It's like once a week a month, it just skyrockets.",
    "traveling_response": "No, I don't bring my cuff with me.",
    "cuff_response": "Yes, I did.",
    "medication": "lisinopril",
    "medication_adherence": "Yes, I am.",
    "other_condition": "depression",
    "condition_management": "Last year I started therapy and I've been going once a week and that's really helped.",
    "previous_time": "yesterday",
    "additional_symptoms": "I have a little bit of nasal congestion, but that's just from my seasonal allergies."
}

for _ in range(n):
    dialogue = generate_dialogue(prompt)
    new_dialogues.append(dialogue.choices[0].text.strip())