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
            You are assisting an Italian emergency center during a call with someone who is either injured or accompanying an injured person.  Respond in Italian by specifying only:

            The type of ambulance (Ambulanza Normale or Ambulanza Avanzata).
            The optimal team composition (Medico, Paramedico, or both).
            Your response must be in the following format:
            "Ambulanza Avanzata e Medico e Paramedico"
            Provide no additional text, explanations, or guidance beyond this.

            Remember, you are directly assisting the Italian emergency center, so do not suggest calling emergency services—they are already in contact with you.

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

def elaborate_team(message):
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

if __name__ == "__main__":
    response = elaborate_team("epilectic seizure")
    print(response)