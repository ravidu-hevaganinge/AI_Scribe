from openai import OpenAI
client = OpenAI()

audio_file= open("C:\\Users\\Ravidu\\Documents\\GitHub\\AI_Scribe\\voice_recording\\SI(ACH000).m4a", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file
)
print(transcription.text)