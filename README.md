# Celsus 118

The proposed model is able to collect information on the patient's emergency status by combining it with information obtained from the patient's Fascicolo Sanitario Elettronico (FSE).

## How to install locally

Start by getting a copy of the github code.
Then in the main project folder, install the requirements:

```bash
  pip install -r requirements.txt
```

Finally, you can run the main script as follows:

```bash
  streamlit run celsus118.py
```

**Important Note**: the system requires some API keys to work. In the main project folder, create a *.env* file, following the structure of *.env.example*:


```bash
GROQ_API_KEY = YOUR-GROQ-API-KEY
AIML_API_KEY = YOUR-AIML-API-KEY
```
