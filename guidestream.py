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
        You're assisting an italian emergency center that is called by a person that is either injured or with the injured person.
        Given an injury or an altered state, give a first aid step-by-step italian manual for that situation.
        The italian manual should include what-ifs situation, so it should take in account the possibility that certain actions are impossible to make or that they could be ineffective.
        The what-ifs should be inluded in each step.
        You are assisting an  italianemergency center, so you shouldn't say to call emergency services since they are already in a call with them.
        Remember, tou should give only the step-by-step italian manual and nothing else!
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

def elaborate_status(message):
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
    response = elaborate_status("epilectic seizure")
    print(response)