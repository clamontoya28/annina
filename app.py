from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
import json
from dotenv import load_dotenv  

# --- Carica variabili d'ambiente ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app, origins=["https://altvera.eu", "https://www.altvera.eu", "http://127.0.0.1:5500/"], methods=["POST"], allow_headers=["Content-Type"]) 

# --- Inizializza OpenAI ---
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
except ImportError:
    print("Errore: libreria OpenAI non installata correttamente.")
    client = None

# --- Carica dataset contestuale (opzionale) ---
try:
    with open('annina_dataset.json', 'r', encoding='utf-8') as f:
        annina_training_data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    annina_training_data = []
    print("Dataset non trovato o danneggiato. L'AI lavorerà senza contesto aggiuntivo.")

# --- Carica memoria emozionale ---
try:
    with open('mini_memoria.json', 'r', encoding='utf-8') as f:
        annina_memory = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    annina_memory = []
    print("Memoria emozionale non trovata o corrotta.")

# --- Endpoint principale ---
@app.route('/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({"error": "Richiesta non in formato JSON"}), 400

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "Il campo 'message' è richiesto"}), 400

    if not OPENAI_API_KEY:
        return jsonify({"error": "Chiave API non configurata"}), 500

    if client is None:
        return jsonify({"error": "Client OpenAI non inizializzato"}), 500

    # --- Controlla se esiste un trigger nella memoria emozionale ---
    for item in annina_memory:
        trigger = item.get("trigger", "").lower()
        if trigger in user_message.lower():
            risposta_personalizzata = item.get("response", "").strip()
            if risposta_personalizzata:
                return jsonify({"response": risposta_personalizzata})

    # --- Se nessun trigger trovato, usa OpenAI ---
    try:
        messages = [
            {
                "role": "system",
                "content": "Sei Annina, un'intelligenza artificiale empatica e gentile. Aiuta l'utente ad affrontare le emozioni offrendo ascolto, conforto e consigli pratici. Non sei un medico o psicologo."
            },
            {"role": "user", "content": user_message}
        ]

        completion = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=messages,
            max_tokens=250,
            temperature=0.7
        )
        reply = completion.choices[0].message.content.strip()
        return jsonify({"response": reply})

    except openai.APIError as e:
        print(f"Errore API: {e}")
        return jsonify({"error": "Errore OpenAI"}), 500
    except Exception as e:
        print(f"Errore generico: {e}")
        return jsonify({"error": "Errore interno"}), 500

# --- Route di base per test Render ---
@app.route('/', methods=['GET'])
def index():
    return "Annina AI è attiva e pronta 💖"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
