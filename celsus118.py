import os  # To interact with the operating system and environment variables.
import streamlit as st  # To create and run interactive web applications directly through Python scripts.
from pathlib import Path  # To provide object-oriented filesystem paths, enhancing compatibility across different operating systems.
from dotenv import load_dotenv  # To load environment variables from a .env file into the system's environment for secure and easy access.
from groq import Groq  # To interact with Groq's API for executing machine learning models and handling data operations.
import json
import string
import audiostream, guidestream, emergencyteam
import asyncio
import time
import pandas as pd
import plotly.graph_objects as go
import time
import numpy as np
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv



fseFilePath = "FSECrop.json"
patientPath = "transcription_response.json"
sbool = True        
#LLAMA MODEL
model = "llama3-70b-8192"
flag_call = True


class GroqAPI:
    """Handles API operations with Groq to generate chat responses."""
    def __init__(self, model_name: str):
        load_dotenv()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name

    def _response(self, message):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            temperature=0,
            max_tokens=4096,
            stream=False,
            stop=None,
        )

    def response_stream(self, messages):
        """Generator che itera sugli stream di risposta e yielda il contenuto."""
        for chunk in self._response(messages):
            # Verifica se il chunk contiene contenuto e yielda solo il testo
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.get('content'):
                yield chunk.choices[0].delta['content']

# Generator to stream responses from the API
    """def response_stream(self, message):        
        for chunk in self._response(message):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content"""

class Score:
    system_prompt = "Write me a score from 1-100 that describes the severity of the status of the patient. Return me just a number."
    
    def __init__(self):
        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "system", "content": self.system_prompt}]

    def add(self, role: str, content: str):
        # Add a message to the chat history
        st.session_state.messages.append({"role": role, "content": content})
    
    def get_response(self, llm):
        # Call the LLM and get a response
        response = llm._response(st.session_state.messages)  # Assuming `response` is synchronous and returns the full text
        # Add assistant's response to the chat history

        #Prendo valore statico della risposta
        assistant_message = response.choices[0].message.content.strip()
        st.session_state.messages.append({"role": "assistant", "content": response})
        return assistant_message  # Return the full response text


def getScore(severity):
    llm = GroqAPI("llama3-70b-8192")  # Initialize your AI model
    score = Score()  # Create a Score instance
    score.add("user", severity)  # Add the user input to the chat history
    response = score.get_response(llm)  # Get the AI response
    #print(response)
    return response  # Return the response



st.html("""<!DOCTYPE html>
<html lang="it">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interfaccia Supporto Operatori 118</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
      body {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f2f5;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        height: 100vh;
      }

      .logo {
        width: 50px;
        height: 50px;
        margin-right: 15px;
      }

      .header span {
        font-size: 1.5em;
        font-weight: 600;
      }

      .container {
        display: flex;
        width: 100%;
        height: 100%;
        padding: 20px;
        gap: 20px;
        box-sizing: border-box;
        overflow: hidden;
      }

      .sidebar {
        width: 280px;
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        padding: 20px;
        box-sizing: border-box;
        flex-shrink: 0;
      }

      .sidebar ul {
        list-style: none;
        padding: 0;
      }

      .sidebar ul li {
        margin-bottom: 25px;
        font-size: 1.1em;
        font-weight: 500;
        display: flex;
        align-items: center;
        color: #333;
        transition: all 0.3s;
      }

      .sidebar ul li i {
        margin-right: 12px;
        font-size: 1.4em;
        color: #007bff;
      }

      .sidebar ul li:hover {
        color: #007bff;
        cursor: pointer;
      }

      .content {
        display: flex;
        flex: 1;
        gap: 20px;
        overflow-y: auto;
        padding-bottom: 20px;
      }

      .col {
        flex: 1;
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        padding: 25px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        gap: 20px;
      }

      .section-title {
        font-weight: 600;
        font-size: 1.4em;
        margin-bottom: 15px;
      }

      input[type="text"],
      input[type="date"],
      textarea {
        width: 100%;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #ddd;
        border-radius: 8px;
        box-sizing: border-box;
        font-size: 1em;
        transition: border-color 0.3s;
      }

      input[type="text"]:focus,
      input[type="date"]:focus,
      textarea:focus {
        border-color: #007bff;
      }

      #speechToText {
        height: 120px;
      }

      .checkbox-consent {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
      }

      .checkbox-consent input {
        margin-right: 10px;
      }

      button {
        padding: 14px 24px;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s;
        font-size: 1em;
        font-weight: 500;
      }

      button:hover {
        background-color: #0056b3;
      }

      button:disabled {
        background-color: #ccc;
        cursor: not-allowed;
      }

      #riskIndicator {
        width: 100%;
        height: 40px;
        background: linear-gradient(to right, green, yellow, red);
        border-radius: 10px;
        margin-bottom: 20px;
      }

      .risk-level {
        text-align: center;
        margin-bottom: 10px;
        font-weight: bold;
      }

      .chart-container {
        position: relative;
        height: 300px;
      }

      .medication-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
      }

      .medication-table th,
      .medication-table td {
        border: 1px solid #ccc;
        padding: 12px;
        text-align: left;
        font-size: 0.95em;
      }

      .medication-table th {
        background-color: #f0f2f5;
        font-weight: 600;
      }
        
      .st-emotion-cache-12fmjuu {
        display: none;
      }
        
      .st-emotion-cache-h4xjwg {
        display: none;
      }
    </style>
  </head>""")

# CSS per il layout
st.markdown("""
    <style>
    @keyframes fillAnimation {
    from {
        background: conic-gradient(transparent 0%, #e9ecef 0% 100%);
    }
    to {
        background: conic-gradient(var(--color) 0% var(--percentage)%, #e9ecef var(--percentage)% 100%);
    }
    }
    /* Rimuove lo stile dei bottoni nel menu */
    [data-testid="stHorizontalBlock"] > div:first-child button {
        background: none;
        border: none;
        padding: 0;
        margin: 0;
        font-weight: normal;
        color: #333333;  /* Colore nero per i link */
        text-align: left;
        width: 100%;
    }
    
    [data-testid="stHorizontalBlock"] > div:first-child button:hover {
        color: #000000;
        text-decoration: underline;
    }

    /* Layout a schermo intero */
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* Header style */
    .header-container {
        border-bottom: 1px solid #e5e5e5;
        margin-top: -15px;
        margin-bottom: 30px;
        text-align: left;
        margin-left: 30px;
        margin-right: 30px;
    }

    /* Colonne style */
    [data-testid="stHorizontalBlock"] {
        padding: 0 1rem;
        gap: 1rem;
    }

    /* Nasconde il titolo del menu */
    [data-testid="stHorizontalBlock"] > div:first-child h3 {
        display: none;
    }

    /* Aggiunge margin-top al menu */
    [data-testid="stHorizontalBlock"] > div:first-child {
        margin-top: 50px;
    }

    /* Stile per i titoli delle colonne */
    [data-testid="stHorizontalBlock"] > div h3 {
        font-size: 25px !important;
        font-weight: 600 !important;
        margin-bottom: 20px !important;
    }

    /* Divider verticale dopo il menu */
    [data-testid="stHorizontalBlock"] > div:first-child {
        border-right: 1px solid #e5e5e5;
        padding-right: 2rem;
        margin-right: 1rem;
    }

    /* Riduzione dimensioni elementi nelle colonne */
    [data-testid="stHorizontalBlock"] > div:not(:first-child) {
        font-size: 0.8em;
    }
    
    
    /* Riduzione dimensioni slider */
    [data-testid="stHorizontalBlock"] > div:not(:first-child) .stSlider {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Riduzione dimensioni tabella */
    [data-testid="stHorizontalBlock"] > div:not(:first-child) .stTable {
        font-size: 0.8em !important;
    }
    
    /* Riduzione dimensioni grafico */
    [data-testid="stHorizontalBlock"] > div:not(:first-child) .js-plotly-plot {
        height: 320px !important;
    }
    
    /* Riduzione padding generale */
    [data-testid="stHorizontalBlock"] > div:not(:first-child) {
        padding: 0.8em !important;
    }
    
    /* Riduzione spazio tra elementi */
    [data-testid="stHorizontalBlock"] > div:not(:first-child) > div {
        margin-bottom: 0.8em !important;
    }

    /* Stile per l'area scrollabile */
    .scrollable-text {
        height: 400px;
        overflow-y: auto;
        padding: 15px;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        background-color: #ffffff;
    }

    /* Personalizzazione della scrollbar */
    .scrollable-text::-webkit-scrollbar {
        width: 8px;
    }

    .scrollable-text::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }

    .scrollable-text::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }

    .scrollable-text::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    </style>
""", unsafe_allow_html=True)

# Header con logo e titolo
st.logo("logo.png", size="large")
st.markdown("""
    <div class="header-container">
        <h2></h2>
    </div>
""", unsafe_allow_html=True)

# Layout a 4 colonne che coprono tutta la larghezza
menu_col, col1, col2, col3 = st.columns([0.5, 1, 1, 1])

# Menu nella prima colonna con icone moderne
with menu_col:
    st.markdown("### Menu")
    menu_items = {
        "⚡ Dashboard": "dashboard",
        "📱 Chiamate": "calls",
        "🎯 Interventi": "interventions",
        "📊 Analytics": "analytics",
        "⚙️ Impostazioni": "settings",
        "📝 Report": "reports",
        "🔍 Ricerca": "search",
    }
    
    # Sostituisci i bottoni con testo stilizzato
    for label in menu_items:
        st.markdown(f"""
            <div class="menu-item">
                {label}
            </div>
        """, unsafe_allow_html=True)

# Aggiungi questo CSS agli stili esistenti
st.markdown("""
    <style>
    /* ... altri stili ... */
    
    .menu-item {
        padding: 8px 16px;
        margin: 4px 0;
        color: #333333;
        cursor: default;
        font-size: 16px;
        transition: background-color 0.3s;
    }
    
    .menu-item:hover {
        background-color: #f0f2f5;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Aggiungi questo CSS per lo stile della proposta di intervento
st.markdown("""
    <style>
    /* ... altri stili esistenti ... */
    
    .team-proposal {
        background-color: #f8f9fa;
        border-radius: 12px;
        margin-top: -15px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .team-proposal-header {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        color: #1a73e8;
        font-weight: 600;
        font-size: 20px;
    }
    
    .team-proposal-content {
        background-color: white;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #1a73e8;
        font-size: 16px;
    }
    
    .section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 16px;
        font-weight: 600;
    }
    
    .section-header i {
        font-size: 1.2em;
    }
    </style>
""", unsafe_allow_html=True)

# Modifica le intestazioni delle sezioni nella colonna 2
with col2:
    st.markdown(f"""### Rischio""")
    
    # Indice di Rischio
    rischio_indice = st.markdown("Indice di rischio generale:")
   
    codicecolore = st.markdown("""
        <div style='background-color: grey; color: white; 
                    padding: 0.5rem; border-radius: 8px; text-align: center;'>
            Codice di gravità Proposto: 
        </div><br>
    """, unsafe_allow_html=True)
    st.markdown("---")  # Separatore
    # Proposta di Intervento
    st.markdown(f"""### Proposta di intervento""")
    team = st.markdown("\n")  # Placeholder che verrà aggiornato
    st.markdown("---")  # Separatore
    st.markdown(f"""### Storico farmaci (da FSE)""")
    
    farmaci = st.markdown("Farmaci trovati: Non disponibile o Non acconsentito")
    # Codice Gravità
    
    
    

with col3:
    st.markdown(f"""### Generazione consigli di primo soccorso""")
    
    guide = st.markdown("""
        <div class="scrollable-text">
            Istruzioni primo soccorso:
        </div>
    """, unsafe_allow_html=True)
    


def getFarmaciList(name):
    with open(fseFilePath) as f:
        for object in json.load(f):
            if object["nome"] == name:
                return " ".join(object["farmaci"])
    return ""

def checkInfo():
    with open(patientPath, 'r') as json_file:
            data = json.load(json_file)
    name = data.get("name", None)  # Usa "Nome" se il campo è scritto in italiano
    status = data.get("status")
    print(name)
    farmaco = getFarmaciList(name)
    criticita = 0
    if farmaco != "" and data.get("consent") != "False":
        print(farmaco)
        severity = status+farmaco
        criticita = getScore(severity)
        print("Cricità di "+name+" Farmaco "+farmaco+" Valore : " + criticita)
        farmaci.markdown(f"Farmaci trovati: {farmaco}")

    else:
        criticita = getScore(status)
        print("Cricità di "+name+" Valore : " + criticita)
    #farmaci.write(f"Farmaci trovati: {farmaco}")
    number_int = int(criticita)
    print(criticita)
    if number_int <= 30:
        rischio_indice.markdown(f"Indice di rischio generale: {number_int}% ")
        codicecolore.markdown("""
        <div style='background-color: green; color: white; 
                    padding: 0.5rem; border-radius: 8px; text-align: center;'>
            Codice di gravità Proposto: Verde
        </div>
    """, unsafe_allow_html=True)
    elif number_int >= 75:
        rischio_indice.markdown(f"Indice di rischio generale: {number_int}% ")
        codicecolore.markdown("""
        <div style='background-color: red; color: white; 
                    padding: 0.5rem; border-radius: 8px; text-align: center;'>
            Codice di gravità Proposto: Rosso
        </div>
    """, unsafe_allow_html=True)

    else:
        rischio_indice.markdown(f"Indice di rischio generale: {number_int}% ")
        codicecolore.markdown("""
        <div style='background-color: yellow; color: black; 
                    padding: 0.5rem; border-radius: 8px; text-align: center;'>
            Codice di gravità Proposto: Giallo
        </div>
    """, unsafe_allow_html=True)
    guide.markdown(f"""
        <div class="scrollable-text">
            {guidestream.elaborate_status(status)}
        </div>
    """, unsafe_allow_html=True)
    # Aggiorna la visualizzazione del team con il nuovo stile
    team_content = emergencyteam.elaborate_team(status)
    team.markdown(f"""
        <div class="team-proposal">
            <div class="team-proposal-header">
                <i class="fas fa-users"></i>
                &nbsp;Team Proposto
            </div>
            <div class="team-proposal-content">
                {team_content}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
if "sbool" not in st.session_state:
    st.session_state.sbool = True
else:
    if not st.session_state.sbool:
        st.session_state.inCall = False
        checkInfo()

if "inCall" not in st.session_state:
    st.session_state.inCall = False

if "name" not in st.session_state:
    st.session_state.name = ''
if "location" not in st.session_state:
    st.session_state.location = ''
if "status" not in st.session_state:
    st.session_state.status = ''
if "consent" not in st.session_state:
    st.session_state.consent = False

def parse_response(response):
    try:
        data = json.loads(response)
        name = data.get('name', 'Non disponibile')
        location = data.get('location', 'Non disponibile')
        status = data.get('status', 'Non disponibile')
        consent = data.get('consent', 'Non disponibile')
        return name, location, status, consent
    except json.JSONDecodeError:
        # Gestisci il caso in cui la risposta non sia un JSON valido
        return "Non disponibile", "Non disponibile", "Non disponibile", "Non disponibile"

def setSbool():
    st.session_state.sbool = False
    audiostream.setStreamBool(st.session_state.sbool)

# Prima aggiungiamo le variabili di stato per i dati della chiamata
if "current_call_data" not in st.session_state:
    st.session_state.current_call_data = {
        "nome": "In attesa...",
        "posizione": "In attesa...",
        "stato": "In attesa...",
        "consenso": "In attesa..."
    }

# Modifichiamo la funzione updateData per aggiornare direttamente i dati visualizzati
async def updateData():
    global sbool
    dati_container = st.empty()
    
    if st.button("Conferma Dati", on_click=setSbool):
        st.session_state.sbool = False
        return

    while sbool:
        await asyncio.sleep(5)
        with open(patientPath, 'r') as json_file:
            data = json.load(json_file)
        
        if "name" in data:
            st.session_state.current_call_data['nome'] = data['name']
        if "location" in data:
            st.session_state.current_call_data['posizione'] = data['location']
        if "status" in data:
            st.session_state.current_call_data['stato'] = data['status']
        if "consent" in data:
            st.session_state.current_call_data['consenso'] = data['consent']
        
        # Aggiorna la visualizzazione dei dati
        dati_container.markdown(f"""
            Nome: {st.session_state.current_call_data['nome']}  
            Posizione: {st.session_state.current_call_data['posizione']}  
            Stato: {st.session_state.current_call_data['stato']}  
            Consenso: {st.session_state.current_call_data['consenso']}
        """)

async def avviastream():
    # Avvia streamaudio e timestop in parallelo
    stream_task = asyncio.create_task(audiostream.streamaudio())
    updateGUI = asyncio.create_task(updateData())
    await asyncio.gather(stream_task,updateGUI)
    #PRESI I DATI


    
with col1:
    # Container per i dati della chiamata
    dati_container = st.empty()
    
    # Mostra i dati correnti
    dati_container.markdown(f"""
        ### Dati Chiamata
        Nome: {st.session_state.current_call_data['nome']}  
        Posizione: {st.session_state.current_call_data['posizione']}  
        Stato: {st.session_state.current_call_data['stato']}  
        Consenso: {st.session_state.current_call_data['consenso']}
    """)
    
    st.markdown("---")  # Separatore
    
    if not st.session_state.inCall:
        chiamata = st.markdown("### 🔴 In Attesa chiamata")

        if st.button("Avvia Chiamata"):
            # Reset dei dati per la nuova chiamata
            st.session_state.current_call_data = {
                "nome": "In attesa...",
                "posizione": "In attesa...",
                "stato": "In attesa...",
                "consenso": "In attesa..."
            }
            chiamata.markdown("### 🟢 In Chiamata")
            st.session_state.inCall = True
            asyncio.run(avviastream())
    else:
        chiamata = st.markdown("### 🟢 In Chiamata")

    


# Footer
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white;
        padding: 1rem;
        border-top: 1px solid #e5e5e5;
        font-size: 0.8em;
        color: #666;
        text-align: center;
        z-index: 999;
    }
    
    .footer-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    .footer-links a {
        color: #666;
        text-decoration: none;
        margin: 0 1rem;
    }
    
    .footer-links a:hover {
        color: #333;
        text-decoration: underline;
    }
    
    .st-emotion-cache-18ni7ap {
        display: none;
    }
           
    .st-emotion-cache-1avcm0n {
        display: none;
    }
    </style>
    
    <div class="footer">
        <div class="footer-content">
            <div class="footer-copyright">
                © 2024 CELSUS118. Tutti i diritti riservati.
            </div>
            <div class="footer-links">
                <a href="#">Privacy Policy</a>
                <a href="#">Termini di Servizio</a>
                <a href="#">Contatti</a>
            </div>
            <div class="footer-info">
                Versione 1.0.0 | Sviluppato da Capyteam
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)





