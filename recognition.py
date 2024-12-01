from groq import Groq  # To interact with Groq's API for executing machine learning models and handling data operations.
from dotenv import load_dotenv
import os

class GroqAPI:
    """Handles API operations with Groq to generate chat responses."""

    def __init__(self, model_name: str):
        load_dotenv()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name

    def _response(self, message):
        """Internal method to fetch responses from the Groq API."""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            temperature=0,
            max_tokens=4096,
            stream=True,
            stop=None,
        )

    def response_stream(self, message):
        """Generator to stream responses from the API."""
        for chunk in self._response(message):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class Message:
    """Manages chat messages."""

    system_prompt = (
        """
        Extract the name of the injured, it's location, the injury and the consent to access its informations. 
        The user message is a conversation between multiple parties.
        Your output should be a JSON with these fields:    
        {
			"name" : "HE NAME OF THE INJURED",
            "location" : "THE LOCATION OF THE INJURED",
            "status" : "THE STATUS OR INJURY OF THE INJURED",
            "consent" : "IF THE INJURED GAVE CONSENT TO ACCESS ITS DATA",
        } 
        The consent should be a Boolean, True if it give consent, False if it doesn't.
        You should understand the name of the injured, which could be or not the interlocutor of the call! 
        There could be cases where there aren't enough datas, just write only the ones you find.
        Here's some examples:
        User message: 'Hi 118, hi, I'm in trouble, I accidentaly hurt my head and I'm bleeding, ok, keep calm, where are you? I'm in Via del Corso 7, ok noted, what's your name? mario rossi, do you give us consent to use your sanitary informations? yes'
        {
			"name" : "Mario Rossi",
            "location" : "Via del Corso 7",
            "status" : "Head emorragy after a hit",
            "consent" : "True"
        }
        User message: 'Hi 118, hi, I'm Mariana Giusti, my father just collapsed after his heart started hurting, ok, what's your father name? Angelo Giusti. where are you? I'm in piazza como, ok noted, do you give us consent to use your father's sanitary informations? no'
        {
			"name": "Angelo Giusti",
            "location" : "Piazza Como",
            "status" : "Possible heart attack",
            "consent" : "False"
        }
        User message: 'buongiorno 118, sono Mario Verdi, sono caduto e il braccio mi si è gonfiato, dove si trova? a via Ugo Ojetti 9, da il consenso per il fascicolo sanitario? sì'
        {
        	"name": "Mario Verdi",
            "location" : "Via Ugo Ojetti 8",
            "status" : "Frattura di un braccio",
            "consent" : "True"
        }
        Use the language that you receive in input from the user!
        Remember to just write the JSON and nothing else.
        """

    )

    def __init__(self, message):
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": message},
        ]

    def add(self, role: str, content: str):
        """Add a new message."""
        self.messages.append({"role": role, "content": content})

    def get_chat_history(self):
        """Retrieve chat history."""
        return self.messages

class ModelSelector:
    """Allows the selection of a model from a predefined list."""

    def __init__(self):
        self.models = [
            "llama3-70b-8192",
        ]

    def select(self, index=0):
        """Select a model by index (default is the first model)."""
        if 0 <= index < len(self.models):
            return self.models[index]
        raise ValueError("Invalid model index")

def elaborate_message(message):
    """Main logic for handling API interactions."""
    # Select model
    model_selector = ModelSelector()
    selected_model = model_selector.select()  # Default model selected

    # Initialize message manager
    message_manager = Message(message=message)

    # Instantiate API client
    llm = GroqAPI(selected_model)

    # Get chat history
    chat_history = message_manager.get_chat_history()

    # Stream response from API
    response = "".join(llm.response_stream(chat_history))
    message_manager.add("assistant", response)
    return response

