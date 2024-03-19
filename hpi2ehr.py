import requests
import json

class EpicAPI:
    def __init__(self, base_url, client_id, client_secret):
        self.base_url = base_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = self.get_access_token()

    def get_access_token(self):
        token_url = f"{self.base_url}/oauth2/token"
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        response = requests.post(token_url, headers=headers, data=data)
        token = response.json().get('access_token')
        return token

    def create_encounter(self, patient_id):
        encounter_url = f"{self.base_url}/Encounter"
        headers = {'Authorization': f"Bearer {self.token}"}
        data = {
            'subject': {'reference': f"Patient/{patient_id}"},
            # Add more encounter details as needed
        }
        response = requests.post(encounter_url, headers=headers, json=data)
        encounter_id = response.json().get('id')
        return encounter_id

    def upload_hpi_to_encounter(self, encounter_id, hpi_data):
        hpi_url = f"{self.base_url}/Encounter/{encounter_id}/DocumentReference"
        headers = {'Authorization': f"Bearer {self.token}", 'Content-Type': 'application/fhir+json'}
        response = requests.post(hpi_url, headers=headers, json=hpi_data)
        return response.status_code


# Example usage
if __name__ == "__main__":
    epic_api = EpicAPI(base_url="your_epic_fhir_api_url", client_id="your_client_id", client_secret="your_client_secret")

    # Assume you have patient_id and hpi_data
    patient_id = "patient_id_here"
    hpi_data = {
        # Construct FHIR-compliant HPI data
    }

    encounter_id = epic_api.create_encounter(patient_id)
    status_code = epic_api.upload_hpi_to_encounter(encounter_id, hpi_data)

    if status_code == 201:
        print("HPI uploaded successfully.")
    else:
        print("Failed to upload HPI.")
