import pyaudio
import threading
import wave
import requests
from queue import Queue
from recognition import elaborate_message
import os
import random
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Replace with your actual API key
API_KEY = os.getenv("AIML_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
URL = "https://api.aimlapi.com/stt"
global stream_bool 
stream_bool = True

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5  # Adjust the duration of audio chunks sent for transcription
temp_files = []

def send_audio_for_transcription(audio_data):
	"""
	Send audio data to the transcription API.
	"""
	try:
		files = {'audio': ('audio.wav', audio_data, 'audio/wav')}
		data = {
			"model": "#g1_nova-2-general",
			"language": "it-IT"
		}
		response = requests.post(URL, files=files, data=data, headers=HEADERS)

		if response.status_code >= 400:
			print(f"Error: {response.status_code} - {response.text}")
		else:
			response_data = response.json()
			transcript = response_data["results"]["channels"][0]["alternatives"][0]["transcript"]
			print("[Transcription]", transcript)
			return transcript
	except Exception as e:
		print(f"Error during transcription: {e}")
		return None

def transcription_thread(queue):
	"""
	Handle transcription in a separate thread.
	"""
	global stream_bool
	response = {
		"name": '',
		"location": '',
		"status": '',
		"consent": "False"
	}
	with open('transcription_response.json', 'w') as json_file:
		json.dump([], json_file)  # Inizializza il file con una lista vuota
		
	transcript = ''
	while stream_bool:
		audio_data = queue.get()  # Wait for audio data to be put into the queue
		if audio_data is None:
			break  # Stop the thread when None is received
		
		with open(audio_data, 'rb') as audio_file:
			result = send_audio_for_transcription(audio_file)
			if result:
				transcript += result
				response = elaborate_message(transcript)
				print(response)
				jsondump = json.loads(response)
				with open('transcription_response.json', 'w') as json_file:
					json.dump(jsondump, json_file, indent=4)
					print(f"Risposta sovrascritta nel JSON: {response}")
		os.remove(audio_data)


def record_and_transcribe():
	"""
	Record audio from the microphone and send it for transcription in chunks.
	"""
	global stream_bool
	rand_number = random.randint(0, 1023)
	audio = pyaudio.PyAudio()
	stream = audio.open(format=FORMAT, channels=CHANNELS,
						rate=RATE, input=True,
						frames_per_buffer=CHUNK)
	# Queue for passing audio data to the transcription thread
	queue = Queue()
	
	# Start the transcription thread
	transcription_thread_instance = threading.Thread(target=transcription_thread, args=(queue,))
	transcription_thread_instance.daemon = True  # Ensure thread exits when the program stops
	transcription_thread_instance.start()
	i = 0
	print("Recording...")
	while stream_bool:
		frames = []
		
		for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
			data = stream.read(CHUNK)
			frames.append(data)
		
		# Save the audio data to a WAV file (in-memory)
		wav_data = b''.join(frames)
		temp_file_name = "temp_" + str(rand_number) + "_" + str(i) + ".wav"
		with wave.open(temp_file_name, 'wb') as wf:
			wf.setnchannels(CHANNELS)
			wf.setsampwidth(audio.get_sample_size(FORMAT))
			wf.setframerate(RATE)
			wf.writeframes(wav_data)
		temp_files.append(temp_file_name)
		queue.put(temp_file_name)
		i = i + 1

	stream.stop_stream()
	stream.close()
	audio.terminate()
	queue.put(None)  # Stop the transcription thread

def clean_up_temp_files():
    """
    Delete all temporary files on keyboard interrupt.
    """
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            print(f"Deleted {temp_file}")
        except Exception as e:
            print(f"Failed to delete {temp_file}: {e}")

def setStreamBool(sbool):
  global stream_bool
  stream_bool = sbool

async def streamaudio():
	global stream_bool
	# Use a separate thread for audio recording and transcription
	transcription_thread_instance = threading.Thread(target=record_and_transcribe)
	transcription_thread_instance.daemon = True  # Ensure thread exits with the program
	transcription_thread_instance.start()

	try:
		while stream_bool:
			#print(stream_bool)
			await asyncio.sleep(1)
		clean_up_temp_files()
		print("\nStopping transcription.")
	except KeyboardInterrupt:
		clean_up_temp_files()
		print("\nStopping transcription.")
