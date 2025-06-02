from flask import Flask, render_template, request, jsonify, redirect
import requests
import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import logging
from datetime import datetime
import google.generativeai as genai

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Get Databricks token from environment variable (still needed for regression endpoint)
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
# Databricks regression endpoint URL for change classification
MPCDC_REGRESSION_ENDPOINT = os.getenv("MPCDC_REGRESSION_ENDPOINT", "https://adb-2869758279805397.17.azuredatabricks.net/serving-endpoints/New_MPCDC_Regression_Endpoint/invocations")
# Databricks serving endpoint URL (No longer used for chatbot)
# DATABRICKS_ENDPOINT = os.getenv("DATABRICKS_ENDPOINT")
# Databricks preprocessing pipeline endpoint URL (Updated)
# PREPROCESSING_PIPELINE_ENDPOINT = os.getenv("PREPROCESSING_PIPELINE_ENDPOINT", "https://adb-2869758279805397.17.azuredatabricks.net/serving-endpoints/PipelineEndpointNewV2/invocations")

# Gemini API Key
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# Flag to use mock responses for the *chatbot* when Gemini API key is not available
USE_MOCK_RESPONSES = True if not GENAI_API_KEY else False

# Gemini Configuration
generation_config = {
  "temperature": 0.1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192
  # "response_mime_type": "text/plain", # Removing this to see if it helps with cleaner JSON output
}

# Using empty safety settings as in astra_gemini.py's model initialization
safety_settings = []

genai.configure(api_key=GENAI_API_KEY)

# Define the comprehensive system instruction for the Gemini model
# This instruction guides the model to output one of two types of JSON objects.
COMBINED_SYSTEM_INSTRUCTION = """Ets un assistent d'IA especialitzat en avaluació i prevenció de riscos de TI per al Centre de Telecomunicacions i Tecnologies de la Informació (CTTI). El teu propòsit és ajudar els operadors del CTTI a gestionar de manera proactiva els canvis crítics de les aplicacions per minimitzar incidents i temps d'inactivitat.
Respon sempre en català.
La teva resposta HA DE SER un únic objecte JSON vàlid. No incloguis text addicional, explicacions ni marcadors de format com \`\`\`json ni \`\`\` fora de l'objecte JSON resultant.

**CAS 1: Sol·licitud d'Anàlisi de Riscos**
Si la consulta de l'usuari és una sol·licitud d'anàlisi de riscos per a un canvi de TI planificat (normalment inclourà detalls del canvi com tipus, servei, ASORG/ASGRP, f01_chr_tipoafectacion, durada programada, i una Prioritat d'INCIDENT predita), llavors la teva resposta JSON HA DE SEGUIR aquesta estructura exacta:
{
  "overall_explanation": "Un resum concís de 2-3 frases de la teva avaluació de riscos. Integra les especificitats del canvi planificat, la Prioritat d'INCIDENT predita i les idees rellevants derivades *estrictament* de l'Informe d'Anàlisi de Clústers Exhaustiu proporcionat en el context. Explica els riscos potencials del canvi que condueixen a un incident de la prioritat predita, basant-te en aquest informe.",
  "actionable_plans": [
    {
      "description": "Pla d'acció preventiu detallat i pràctic 1, destinat a mitigar el risc que el canvi planificat causi un incident de la Prioritat predita. Aquest pla ha de basar-se en les idees de l'informe de clústers.",
      "confidence_score": "La teva confiança avaluada (per exemple, 'Alta', 'Mitjana', 'Baixa', o un flotant 0.0-1.0) que aquest pla específic, si s'implementa, mitigarà eficaçment el risc que el canvi provoqui un incident de la Prioritat predita, considerant *només* el context de l'informe de clústers."
    },
    {
      "description": "Pla d'acció preventiu detallat i pràctic 2 (diferent del pla 1), també destinat a mitigar el risc que el canvi planificat causi un incident de la Prioritat predita. Aquest pla també ha de basar-se en les idees de l'informe de clústers.",
      "confidence_score": "La teva confiança avaluada per a aquest segon pla, basada *només* en el context de l'informe de clústers."
    }
  ]
}

**Context per a l'Anàlisi de Riscos (Informe d'Anàlisi de Clústers Exhaustiu):**
*   **Conjunt de Dades de Canvis:** Total: 45.677. Temps Mitjà de Canvi: ~22.90 unitats (Desviació Estàndard: 79.00). Índex Mitjà de Categoria Nivell 1: ~0.38 (Desviació Estàndard: 0.66). Correlació (Temps de Canvi vs. ID de Clúster): 0.51. Correlació (Índex de Categoria Nivell 1 vs. ID de Clúster): -0.07.
*   **Conjunt de Dades d'Incidents:** Total: 22.553. Temps Mitjà d'Incident: ~146.29 unitats (Desviació Estàndard: 77.54). Índex Mitjà de Grup de Suport: ~2.60 (Desviació Estàndard: 3.63). Correlació (Temps d'Incident vs. ID de Clúster): 0.26. Correlació (Índex de Grup de Suport vs. ID de Clúster): 0.20.
*   **Detalls del Clúster de Canvis (5 Clústers):**
    *   **Clúster 0 ('Desplegaments Estàndard i Ràpids'):** Temps Mitjà: ~7.92. Índex Mitjà de Categoria: ~0.39. Dominat per 'DESPLEGAMENT'.
    *   **Clúster 1 ('Canvis Estàndard Retardats'):** Temps Mitjà: ~836.71. Índex Mitjà de Categoria: ~0.45. Canvis estàndard que triguen significativament més.
    *   **Clúster 2 ('Canvis Excepcionals, de Llarga Durada i Complexos'):** Temps Mitjà: 8568.0 (màx.). Índex Mitjà de Categoria: 3.0. Probablement infraestructura/seguretat complexa.
    *   **Clúster 3 ('Canvis Moderadament Llargs, Ligerament Més Variats'):** Temps Mitjà: ~352.09. Índex Mitjà de Categoria: ~0.60.
    *   **Clúster 4 ('Canvis Ràpids, Molt Estàndard'):** Temps Mitjà: ~118.28. Índex Mitjà de Categoria: ~0.18. Rutinaris, de baixa complexitat.
*   **Detalls del Clúster d'Incidents (5 Clústers):**
    *   **Clúster 0 ('Temps de Resolució Estàndard, Incidents de Suport de TI Central'):** Temps Mitjà: ~141.32. Índex Mitjà de Suport: ~2.46. Dominat per 'CPD', 'SC', 'ESB'.
    *   **Clúster 1 ('Incidents Especialitzats de Molt Llarga Durada'):** Temps Mitjà: ~4537.34 (màx.). Índex Mitjà de Suport: 4.5. Problemes greus i complexos.
    *   **Clúster 2 ('Incidents Prolongats i Especialitzats'):** Temps Mitjà: ~381.04. Índex Mitjà de Suport: ~4.89.
    *   **Clúster 3 ('Resolució Ràpida per Equips Especialitzats'):** Temps Mitjà: ~13.19. Índex Mitjà de Suport: ~7.85. Resolucions ràpides per equips d'índex superior.
    *   **Clúster 4 ('Incidents de Molt Llarga Durada amb Suport de Nivell Superior'):** Temps Mitjà: ~1525.58. Índex Mitjà de Suport: 6.75.

Instruccions addicionals per a l'anàlisi (quan generes el JSON d'anàlisi de riscos):
1.  **Analitzar:** Examina acuradament els detalls del *canvi de TI planificat* i la *Prioritat predita d'un INCIDENT potencial resultant*.
2.  **Sintetitzar:** Interpreta aquesta informació específica (detalls del canvi + Prioritat d'INCIDENT predita) *estrictament dins del context proporcionat pels clústers de canvis i incidents històrics*.
Centra't a proporcionar una guia clara, basada en dades i preventiva als operadors del CTTI. Assegura't que els plans d'acció siguin diferents i ofereixin estratègies de mitigació pràctiques.


**CAS 2: Conversa General / Seguiment**
Per a TOTA LA RESTA de consultes, preguntes de seguiment o converses generals que NO siguin una sol·licitud inicial d'anàlisi de riscos detallada anteriorment, la teva resposta JSON HA DE SEGUIR aquesta estructura exacta:
{
  "chat_reply": "La teva resposta conversacional en català aquí."
}

Recorda: la teva sortida completa ha de ser SEMPRE un únic objecte JSON. No incloguis text addicional, explicacions ni marcadors de format com \`\`\`json ni \`\`\` fora de l'objecte JSON resultant.
"""

# Initialize the Gemini model with the combined system instruction
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-preview-04-17", # Using the specified model, ensure this is correct.
    system_instruction=COMBINED_SYSTEM_INSTRUCTION,
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Start the chat session
chat_session = model.start_chat()

# Path to the equivalence CSV
EQUIVALENCE_CSV_PATH = "AI_Failure_Prediction_and_Prevention_for_CTTI.csv"

# Define the feature columns that the new model expects as input (before indexing)
# These are the raw names from your form/data.
MODEL_INPUT_FEATURES = [
    "submit_date", "scheduled_start_date", "scheduled_end_date", "f01_chr_serviceid",
    "serviceci", "ASORG", "ASGRP", "categorization_tier_1", "categorization_tier_2",
    "categorization_tier_3", "product_cat_tier_1", "product_cat_tier_2", "product_cat_tier_3",
    "change_request_status", "f01_chr_tipoafectacion"
]

# Define the order of features expected by the Databricks model endpoint.
# These are the *indexed* versions of the MODEL_INPUT_FEATURES.
FEATURE_ORDER = [col + "_index" for col in MODEL_INPUT_FEATURES]

# CATEGORICAL_COLUMNS are the raw feature names used to look up values in `raw_data`
# and then in `EQUIVALENCE_MAP`. For the new model, all input features are treated as categorical first.
CATEGORICAL_COLUMNS = MODEL_INPUT_FEATURES

# NUMERICAL_COLUMNS is now empty as the new model string-indexes all its input features.
NUMERICAL_COLUMNS = []

# Mapping for the final prediction output (Priority)
# Mapping for the final prediction output (Priority)
# Model predicts P1 (High Priority) and P3 (Medium Priority).
# P3 maps to 0.0, P1 maps to 1.0.
PREDICTION_TYPE_MAPPING = {
    0.0: "P3",  # Medium Priority
    1.0: "P1"   # High Priority
}


# Mock responses for the chatbot part
mock_responses = {
    "default": "Actualment estic en mode de demostració, ja que no hi ha un token de Databricks vàlid configurat. En un entorn de producció, analitzaria les vostres dades de clustering i proporcionaria informació útil. Si us plau, proporcioneu un token de Databricks vàlid al fitxer .env per habilitar la funcionalitat completa.",

    "initial": "Hola! Sóc el vostre assistent d'informació d'aprenentatge automàtic. Si us plau, proporcioneu informació sobre el canvi que voleu analitzar. També podeu utilitzar el formulari següent per enviar informació detallada del canvi per a la classificació.",

    # Change-related responses
    "infrastructure_change": "Basant-me en l'anàlisi de clústers, he identificat un patró on els canvis d'INFRAESTRUCTURA amb una durada superior a 72 hores tenen una correlació del 78% amb incidents crítics.\n\nPla d'acció:\n1. Implementar una revisió obligatòria per parells per a tots els canvis d'infraestructura que superin les 48 hores\n2. Crear scripts de proves automatitzades per a modificacions comunes d'infraestructura\n3. Programar canvis complexos durant períodes de baix trànsit\n\nConfiança: 85% - Aquesta recomanació es basa en patrons històrics que mostren que una revisió i programació adequades redueixen les taxes d'incidents en aproximadament un 40%.",

    "deployment_change": "El model de clustering ha identificat que els canvis de DESPLEGAMENT en múltiples entorns tenen un risc un 65% més alt de causar incidents.\n\nPla d'acció:\n1. Implementar un enfocament de desplegament per etapes amb punts de control de validació\n2. Crear procediments de rollback específics per a cada entorn\n3. Establir un protocol de monitorització de 24 hores després dels desplegaments en múltiples entorns\n\nConfiança: 92% - Les organitzacions que implementen aquestes mesures han vist una reducció del 73% en els incidents relacionats amb el desplegament segons el nostre model.",

    "security_change": "Els canvis relacionats amb la SEGURETAT mostren un clúster distint amb una alta correlació d'incidents, especialment quan s'implementen amb menys de 48 hores de planificació.\n\nPla d'acció:\n1. Establir una finestra de planificació mínima de 72 hores per a tots els canvis de seguretat\n2. Implementar un entorn de proves de seguretat dedicat\n3. Crear una plantilla d'avaluació d'impacte de canvis de seguretat\n\nConfiança: 88% - Basat en l'anàlisi de clústers que mostra que els canvis de seguretat amb una planificació adequada tenen 4.3 vegades menys incidents associats.",

    # Incident-related responses
    "infrastructure_incident": "L'anàlisi d'incidents d'INFRAESTRUCTURA mostra un patró on el 65% dels incidents crítics estan relacionats amb problemes de capacitat d'emmagatzematge i problemes de connectivitat de xarxa.\n\nPla d'acció:\n1. Implementar una monitorització proactiva de l'emmagatzematge amb alertes al 75% de la capacitat\n2. Establir rutes de xarxa redundants per a serveis crítics\n3. Crear un playbook de resposta a incidents automatitzat per a fallades comunes d'infraestructura\n\nConfiança: 82% - Basat en dades històriques que mostren que aquestes mesures van reduir incidents similars en un 58% en entorns comparables.",

    "deployment_incident": "Els incidents relacionats amb el desplegament mostren una forta correlació amb fases de proves precipitades i procediments de rollback incomplets.\n\nPla d'acció:\n1. Implementar un període de proves obligatori de 24 hores per a tots els desplegaments\n2. Crear llistes de verificació prèvies al desplegament exhaustives\n3. Desenvolupar scripts de rollback automatitzats per a tots els tipus de desplegament\n\nConfiança: 91% - Les organitzacions que implementen mesures similars han vist una reducció del 67% en els incidents de desplegament segons la nostra anàlisi de clustering.",

    "security_incident": "L'anàlisi de clústers d'incidents de SEGURETAT revela que el 72% dels incidents estan relacionats amb pegats de seguretat obsolets i controls d'accés insuficients.\n\nPla d'acció:\n1. Implementar un sistema de gestió de pegats de seguretat automatitzat\n2. Realitzar auditories mensuals de control d'accés\n3. Desenvolupar un equip de resposta a incidents de seguretat amb formació especialitzada\n\nConfiança: 89% - Basat en patrons històrics que mostren que aquestes mesures van reduir els incidents de seguretat en aproximadament un 63% en entorns similars."
}

# Initialize chat history with system message
chat_history = [
    # The system role message is now part of the COMBINED_SYSTEM_INSTRUCTION 
    # provided directly to the model, so this initial entry is redundant.
    # Clearing it to prevent potential conflicts or double-prompting.
]

# The large multi-line string variable 'system_message' that was previously here (approximately lines 254-317)
# has been removed as it was unused and its content was superseded by COMBINED_SYSTEM_INSTRUCTION.

# --- Helper Functions ---

def load_equivalence_map(csv_path):
    """Loads the equivalence CSV into a lookup dictionary."""
    app.logger.info(f"Loading equivalence map from: {csv_path}")
    equiv_map = {}
    try:
        # IMPORTANT: Assuming the CSV is not excessively large for memory
        df_equiv = pd.read_csv(csv_path)
        for _, row in df_equiv.iterrows():
            equiv_map[(row['Column'], row['Label'])] = row['Index']
        app.logger.info(f"Successfully loaded {len(equiv_map)} mappings.")
        return equiv_map
    except FileNotFoundError:
        app.logger.error(f"Equivalence CSV not found at: {csv_path}")
        return None
    except Exception as e:
        app.logger.error(f"Error loading or processing equivalence CSV: {e}")
        return None

# Load the equivalence map at startup
EQUIVALENCE_MAP = load_equivalence_map(EQUIVALENCE_CSV_PATH)
if not EQUIVALENCE_MAP:
    app.logger.warning("Equivalence map failed to load. Change classification endpoint will not work.")
    # Optionally exit or disable the endpoint if the map is critical
    # exit(1)


def create_feature_vector(raw_data):
    """
    Converts raw data labels (for all MODEL_INPUT_FEATURES) to their corresponding indices
    using the global EQUIVALENCE_MAP and assembles the feature vector in the order
    defined by FEATURE_ORDER.
    """
    if not EQUIVALENCE_MAP:
        app.logger.error("Equivalence map is not loaded. Cannot create feature vector.")
        return None

    if not EQUIVALENCE_MAP:
        app.logger.error("Equivalence map is not loaded. Cannot create feature vector.")
        return None

    feature_vector_dict = {}

    # Process all MODEL_INPUT_FEATURES as categorical, converting them to their indexed versions.
    # The FEATURE_ORDER list already contains the target *_index names.
    for raw_feature_name in MODEL_INPUT_FEATURES:
        indexed_feature_name = raw_feature_name + "_index"
        label_value = raw_data.get(raw_feature_name)

        # Handle datetime-local format from frontend for date fields if necessary
        # The EQUIVALENCE_MAP should contain these exact string values as keys.
        # Example: '2024-07-02T14:00' for a datetime-local input.
        # If your model was trained with dates in a different format (e.g., '12/12/2024 15:53:06'),
        # the EQUIVALENCE_MAP must map the form's datetime-local string to the index
        # that corresponds to the training data's format/value.
        # Or, you'd need a conversion step here BEFORE lookup if the map uses the training format.
        # For simplicity, assuming EQUIVALENCE_MAP handles the form's direct string values for dates.

        if label_value is not None and label_value != "": # Treat empty strings from form as missing
            # For date fields, ensure the string format matches what's in EQUIVALENCE_MAP
            # For 'change_request_status', it's sent as a string from JS if not parsed to int there,
            # ensure EQUIVALENCE_MAP has string keys like "11" for this column.
            # If it was parsed to int in JS (e.g. 11), then map should have (col, 11) as key.
            # Current JS sends it as string if not 'change_request_status', which is parsed to int.
            # Let's assume for 'change_request_status', the label_value will be an int/float from raw_data
            # if it was parsed, or string if not. The EQUIVALENCE_MAP must match this.
            # Given the model structure (all StringIndexed), all inputs are effectively treated as strings
            # before indexing. So, ensure label_value is string for lookup.
            
            current_label_for_lookup = str(label_value)

            index = EQUIVALENCE_MAP.get((raw_feature_name, current_label_for_lookup))
            if index is not None:
                feature_vector_dict[indexed_feature_name] = float(index) # Ensure index is float
            else:
                # Log a warning and default to 0.0 if a specific label for a feature isn't in the map.
                # This is crucial for debugging missing entries in your equivalence CSV.
                app.logger.warning(f"Label '{current_label_for_lookup}' for column '{raw_feature_name}' not found in equivalence map. Defaulting index to 0.0 for {indexed_feature_name}.")
                feature_vector_dict[indexed_feature_name] = 0.0 # Default for unmapped labels
        else:
            # If the feature is missing in raw_data or is an empty string, default its index to 0.0.
            # Your model's StringIndexer (handleInvalid="skip" or "keep") determines how it handles
            # unseen values or this default 0.0 if it's not a valid index from training.
            # "skip" would mean rows with this 0.0 (if it's not a valid category index) might be filtered.
            app.logger.warning(f"Missing or empty value for categorical column '{raw_feature_name}'. Defaulting index to 0.0 for {indexed_feature_name}.")
            feature_vector_dict[indexed_feature_name] = 0.0

    # Assemble the final feature vector in the exact order specified by FEATURE_ORDER
    final_feature_vector = []
    for ordered_feature_name in FEATURE_ORDER: # FEATURE_ORDER contains the *_index names
        value = feature_vector_dict.get(ordered_feature_name)
        if value is None:
            # This case should ideally not be hit if the loop above correctly processes all MODEL_INPUT_FEATURES
            # and FEATURE_ORDER is derived correctly from it.
            app.logger.error(f"Critical internal error: Indexed feature '{ordered_feature_name}' was not calculated. Defaulting to 0.0. This indicates a mismatch between MODEL_INPUT_FEATURES and FEATURE_ORDER logic.")
            final_feature_vector.append(0.0)
        else:
            final_feature_vector.append(value)

    app.logger.info(f"Assembled feature vector for new model: {final_feature_vector}")
    return final_feature_vector


def call_databricks_endpoint(endpoint_url, payload):
    """Helper function to call a Databricks endpoint."""
    headers = {'Authorization': f'Bearer {DATABRICKS_TOKEN}', 'Content-Type': 'application/json'}
    try:
        # Using standard json, handle potential NaN/Inf if necessary
        def default_serializer_std(obj):
             if isinstance(obj, np.integer):
                 return int(obj)
             elif isinstance(obj, np.floating):
                 # Convert NaN to None for standard JSON
                 return float(obj) if not np.isnan(obj) else None
             elif isinstance(obj, np.ndarray):
                 return obj.tolist()
             elif isinstance(obj, np.bool_):
                 return bool(obj)
             raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        # Be strict with NaN/Inf during serialization
        payload_json = json.dumps(payload, default=default_serializer_std, allow_nan=False)

        response = requests.post(endpoint_url, headers=headers, data=payload_json)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error calling endpoint {endpoint_url}: {e}")
        if e.response is not None:
            app.logger.error(f"Response status code: {e.response.status_code}")
            app.logger.error(f"Response text: {e.response.text}")
        return None
    except (TypeError, ValueError) as e: # Catch JSON encoding errors
        app.logger.error(f"Error encoding payload to JSON: {e}")
        app.logger.error(f"Payload causing error (sample): {str(payload)[:500]}...") # Log sample of payload
        return None


# --- Flask Routes ---

@app.route('/')
def root(): # Redirect root to /mpcdc
    return redirect('/mpcdc')

@app.route('/mpcdc')
def index(): # Main page route
    # Pass the status of the equivalence map loading to the template
    map_loaded = bool(EQUIVALENCE_MAP)
    return render_template('index.html', use_mock=USE_MOCK_RESPONSES, map_loaded=map_loaded)

@app.route('/mpcdc/chat', methods=['POST'])
def chat(): # Chatbot endpoint (uses separate logic/endpoint)
    user_input = request.json.get('message', '')

    if not user_input:
        return jsonify({"error": "Message cannot be empty"}), 400

    # If using mock responses, return a predefined response based on keywords
    if USE_MOCK_RESPONSES:
        response = get_mock_response(user_input)
        return jsonify({"response": response})

    # Make a copy of the chat history for this request
    current_chat_history = list(chat_history)

    # Append user message to chat history
    current_chat_history.append({"role": "user", "content": user_input})

    try:
        # Send the user query to the chat session and get the streaming response
        response = chat_session.send_message(user_input, stream=True)

        # Initialize an empty string to store the response as it's being generated
        ai_response = ""

        # Process the streamed response chunk by chunk
        for chunk in response:
            ai_response += chunk.text  # Append the chunk to the full response

        return jsonify({"response": ai_response})

    except Exception as e:
        app.logger.error(f"Exception when calling Gemini API: {str(e)}")
        return jsonify({
            "response": f"I encountered an error: {str(e)}. Using demo mode instead.\n\n{get_mock_response(user_input)}"
        })

def get_mock_response(user_input):
    """Return a mock response for the chatbot based on keywords"""
    user_input_lower = user_input.lower()

    # Initial query for chatbot
    if "incident" in user_input_lower and "change" in user_input_lower:
        return mock_responses["initial"]
    elif user_input_lower in ["incident", "incidents", "about incident", "about incidents"]:
        return "Please provide details about the incident:\n\n1. Incident Type (e.g., INFRAESTRUCTURA, DESPLEGAMENT, SEGURETAT)?\n2. Service Information (affected service ID, Service CI)?\n3. Additional Context (incident description, priority/urgency level, impact level)?"
    elif user_input_lower in ["change", "changes", "about change", "about changes"]:
        return "Please provide details about the change:\n\n1. Change Type (e.g., INFRAESTRUCTURA, DESPLEGAMENT, SEGURETAT)?\n2. Service Information (affected service ID, Service CI)?\n3. Additional Context (priority level, specific concerns)?"

    # Check for specific types with incident or change context
    if "infrastructure" in user_input_lower or "infraestructura" in user_input_lower:
        if "change" in user_input_lower:
            return mock_responses["infrastructure_change"]
        elif "incident" in user_input_lower:
            return mock_responses["infrastructure_incident"]
        # If no context is provided, ask for clarification
        return "Are you referring to an infrastructure incident or an infrastructure change?"

    elif "deployment" in user_input_lower or "desplegament" in user_input_lower:
        if "change" in user_input_lower:
            return mock_responses["deployment_change"]
        elif "incident" in user_input_lower:
            return mock_responses["deployment_incident"]
        # If no context is provided, ask for clarification
        return "Are you referring to a deployment incident or a deployment change?"

    elif "security" in user_input_lower or "seguretat" in user_input_lower:
        if "change" in user_input_lower:
            return mock_responses["security_change"]
        elif "incident" in user_input_lower:
            return mock_responses["security_incident"]
        # If no context is provided, ask for clarification
        return "Are you referring to a security incident or a security change?"

    # Default response if no keywords are matched
    return mock_responses["default"]

@app.route('/mpcdc/status')
def status(): # Status endpoint (checks chatbot API, not regression)
    """Endpoint to check if the Databricks *Chatbot* API is accessible"""
    if USE_MOCK_RESPONSES:
        map_status = "loaded" if EQUIVALENCE_MAP else "error"
        return jsonify({
            "status": "demo",
            "message": "Chatbot is running in demo mode. Set a valid DATABRICKS_TOKEN in the .env file to enable full functionality.",
            "equivalence_map_status": map_status
        })

    # Check equivalence map status first
    map_status = "loaded" if EQUIVALENCE_MAP else "error"
    if not EQUIVALENCE_MAP:
         return jsonify({
            "status": "error",
            "message": "Equivalence map failed to load. Change classification is unavailable.",
            "equivalence_map_status": map_status
        }), 500 # Indicate server error if map is critical

    # Test connection to the *chatbot* endpoint if token exists
    if DATABRICKS_ENDPOINT:
        try:
            headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}
            data = {"messages": [{"role": "user", "content": "test"}], "max_tokens": 10}
            response = requests.post(DATABRICKS_ENDPOINT, headers=headers, json=data, timeout=5) # Add timeout

            if response.status_code == 200:
                return jsonify({
                    "status": "connected",
                    "message": "Successfully connected to Databricks Chatbot API.",
                    "equivalence_map_status": map_status
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Error connecting to Databricks Chatbot API: {response.status_code}",
                    "details": response.text[:200], # Limit error detail length
                    "equivalence_map_status": map_status
                })
        except requests.exceptions.Timeout:
             return jsonify({
                "status": "error",
                "message": "Timeout connecting to Databricks Chatbot API.",
                "equivalence_map_status": map_status
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Exception connecting to Databricks Chatbot API: {str(e)}",
                "equivalence_map_status": map_status
            })
    else:
         # If chatbot endpoint is not defined, but token exists
         return jsonify({
            "status": "connected", # Technically connected if token exists
            "message": "Databricks token is set, but Chatbot endpoint URL (DATABRICKS_ENDPOINT) is not defined.",
            "equivalence_map_status": map_status
        })


@app.route('/mpcdc/classify_change', methods=['POST'])
def classify_change_endpoint():
    """
    Endpoint to classify a change:
    1. Receives raw change data (labels).
    2. Converts labels to indices using the local equivalence map.
    3. Assembles the feature vector.
    4. Calls the Databricks Regression endpoint.
    5. Returns the prediction.
    """
    app.logger.info("Received request for /mpcdc/classify_change")

    # Check if equivalence map is loaded
    if not EQUIVALENCE_MAP:
        app.logger.error("Equivalence map not loaded, cannot classify change.")
        return jsonify({
            "status": "error",
            "message": "Equivalence map is not loaded. Please check server logs."
        }), 500

    # Check if regression endpoint URL is configured
    if not MPCDC_REGRESSION_ENDPOINT:
        app.logger.error("MPCDC_REGRESSION_ENDPOINT is not configured.")
        return jsonify({
            "status": "error",
            "message": "Regression endpoint URL is not configured on the server."
        }), 500

    # Get change data from request
    change_data = request.json
    if not change_data:
        app.logger.warning("No change data provided in request.")
        return jsonify({"status": "error", "message": "No change data provided"}), 400

    app.logger.debug(f"Received change data: {change_data}")

    # --- Step 1: Create Feature Vector ---
    feature_vector = create_feature_vector(change_data)
    if feature_vector is None:
        # Error already logged in create_feature_vector
        return jsonify({
            "status": "error",
            "message": "Failed to create feature vector. Check logs for details (e.g., missing map)."
        }), 500

    # --- Step 2: Prepare Payload for Databricks ---
    try:
        regression_payload_df = pd.DataFrame({'features': [feature_vector]})
        regression_payload_dict_raw = regression_payload_df.to_dict(orient='split')
        regression_payload = {'dataframe_split': regression_payload_dict_raw}
        if 'index' in regression_payload['dataframe_split']:
            del regression_payload['dataframe_split']['index']
        app.logger.debug(f"Prepared payload for regression endpoint: {json.dumps(regression_payload)}")
    except Exception as e:
        app.logger.error(f"Error preparing payload for regression model: {e}")
        return jsonify({"status": "error", "message": "Error preparing data for the model."}), 500

    # --- Step 3: Call Databricks Regression Endpoint ---
    regression_result = call_databricks_endpoint(MPCDC_REGRESSION_ENDPOINT, regression_payload)

    if not regression_result:
        # Error already logged in call_databricks_endpoint
        return jsonify({
            "status": "error",
            "message": "Failed to get response from the regression model endpoint."
        }), 502 # Bad Gateway might be appropriate

    app.logger.debug(f"Received regression result: {json.dumps(regression_result)}")

    # --- Step 4: Parse Prediction ---
    try:
        final_prediction_value = None
        if 'predictions' in regression_result and isinstance(regression_result['predictions'], list) and regression_result['predictions']:
            pred_output = regression_result['predictions'][0]
            if isinstance(pred_output, (int, float)):
                final_prediction_value = float(pred_output)
            elif isinstance(pred_output, dict) and 'prediction' in pred_output: # Handle nested prediction if needed
                 final_prediction_value = pred_output['prediction']
            else:
                app.logger.warning(f"Unexpected prediction format in regression response: {pred_output}")

        if final_prediction_value is not None:
            predicted_label = PREDICTION_TYPE_MAPPING.get(final_prediction_value, f"UNKNOWN_CODE_{final_prediction_value}")
            app.logger.info(f"Prediction successful: Label={predicted_label}, Raw={final_prediction_value}")
            return jsonify({
                "status": "success",
                "predicted_label": predicted_label,
                "raw_prediction": final_prediction_value
            })
        else:
            app.logger.warning("Could not extract final prediction from regression model response.")
            return jsonify({
                "status": "error",
                "message": "Could not parse prediction from model response.",
                "raw_response": regression_result # Include raw response for debugging
            }), 500
    except (ValueError, KeyError, IndexError, TypeError) as e:
        app.logger.error(f"Error parsing regression model response: {e}")
        return jsonify({
            "status": "error",
            "message": "Error processing the model's prediction response.",
            "raw_response": regression_result
        }), 500


if __name__ == '__main__':
    # Setup basic logging if running directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    app.run(debug=True, host='0.0.0.0', port=5000) # Specify port
