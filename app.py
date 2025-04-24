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
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

# Using empty safety settings as in astra_gemini.py's model initialization
safety_settings = []

genai.configure(api_key=GENAI_API_KEY)

# Initialize the Gemini model
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-preview-04-17",
    # system_instruction will be handled in the chat history messages
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Start the chat session
chat_session = model.start_chat()

# Path to the equivalence CSV
EQUIVALENCE_CSV_PATH = "AI_Failure_Prediction_and_Prevention_for_CTTI.csv"

# Define the order of features expected by the regression model endpoint
# Matches the VectorAssembler inputCols: categorical indices + numerical
FEATURE_ORDER = [
    "f01_chr_serviceid_index", "serviceci_index", "ASORG_index", "ASGRP_index",
    "categorization_tier_1_index", "categorization_tier_2_index", "categorization_tier_3_index",
    "product_cat_tier_1_index", "product_cat_tier_2_index", "product_cat_tier_3_index",
    "change_request_status", "change_duration"
]

CATEGORICAL_COLUMNS = [col.replace('_index', '') for col in FEATURE_ORDER if col.endswith('_index')]
NUMERICAL_COLUMNS = [col for col in FEATURE_ORDER if not col.endswith('_index')]

# Mapping for the final prediction output (f01_chr_tipoafectacion)
PREDICTION_TYPE_MAPPING = {
    0.0: "SENSE TALL DE SERVEI",
    1.0: "TALL DE SERVEI",
    2.0: "DEGRADACIO"
    # Add other mappings if the model can output more values
}


# Mock responses for the chatbot part
mock_responses = {
    "default": "I'm currently in demo mode since there's no valid Databricks token configured. In a production environment, I would analyze your clustering data and provide actionable insights. Please provide a valid Databricks token in the .env file to enable full functionality.",

    "initial": "Hello! I'm your ML Insights Assistant. Please provide information about the change you'd like to analyze. You can also use the form below to submit detailed change information for classification.",

    # Change-related responses
    "infrastructure_change": "Based on the clustering analysis, I've identified a pattern where INFRAESTRUCTURA changes with duration over 72 hours have a 78% correlation with critical incidents.\n\nAction Plan:\n1. Implement a mandatory peer review for all infrastructure changes exceeding 48 hours\n2. Create automated testing scripts for common infrastructure modifications\n3. Schedule complex changes during low-traffic periods\n\nConfidence: 85% - This recommendation is based on historical patterns showing that proper review and scheduling reduces incident rates by approximately 40%.",

    "deployment_change": "The clustering model has identified that DEPLOYMENT changes across multiple environments have a 65% higher risk of causing incidents.\n\nAction Plan:\n1. Implement a staged deployment approach with validation checkpoints\n2. Create environment-specific rollback procedures\n3. Establish a 24-hour monitoring protocol after multi-environment deployments\n\nConfidence: 92% - Organizations implementing these measures have seen a 73% reduction in deployment-related incidents according to our model.",

    "security_change": "Security-related changes show a distinct cluster with high incident correlation, particularly when implemented with less than 48 hours of planning.\n\nAction Plan:\n1. Establish a minimum 72-hour planning window for all security changes\n2. Implement a dedicated security testing environment\n3. Create a security change impact assessment template\n\nConfidence: 88% - Based on cluster analysis showing that security changes with proper planning have 4.3x fewer associated incidents.",

    # Incident-related responses
    "infrastructure_incident": "Analysis of INFRAESTRUCTURA incidents shows a pattern where 65% of critical incidents are related to storage capacity issues and network connectivity problems.\n\nAction Plan:\n1. Implement proactive storage monitoring with alerts at 75% capacity\n2. Establish redundant network paths for critical services\n3. Create an automated incident response playbook for common infrastructure failures\n\nConfidence: 82% - Based on historical data showing these measures reduced similar incidents by 58% in comparable environments.",

    "deployment_incident": "Deployment-related incidents show a strong correlation with rushed testing phases and incomplete rollback procedures.\n\nAction Plan:\n1. Implement a mandatory 24-hour testing period for all deployments\n2. Create comprehensive pre-deployment checklists\n3. Develop automated rollback scripts for all deployment types\n\nConfidence: 91% - Organizations implementing similar measures have seen a 67% reduction in deployment incidents according to our clustering analysis.",

    "security_incident": "Security incidents cluster analysis reveals that 72% of incidents are related to outdated security patches and insufficient access controls.\n\nAction Plan:\n1. Implement an automated security patch management system\n2. Conduct monthly access control audits\n3. Develop a security incident response team with specialized training\n\nConfidence: 89% - Based on historical patterns showing these measures reduced security incidents by approximately 63% in similar environments."
}

# Initialize chat history with system message
chat_history = [
    {
        "role": "system",
        "content": ( """
You are an AI assistant specialized in IT risk assessment and prevention for the Centre de Telecomunicacions i Tecnologies de la Informació (CTTI). Your purpose is to help CTTI operators proactively manage critical application changes to minimize incidents and downtime.

You will receive contextual information derived from Machine Learning models trained on historical CTTI data regarding IT changes ('canvis') and incidents ('incidencies'). This context includes:

1.  **Changes Clusters (`canvis_clusters`):**
    *   These clusters group historical IT changes based on characteristics like `Categorization_tier_1` (type of change, e.g., DESPLEGAMENT, INFRAESTRUCTURA, SEGURETAT) and `change_time` (duration of the change window in hours).
    *   You are given `canvis_clusters_summary`: Statistical summaries (count, mean, stddev, min, max) of variables within these change clusters. Pay attention to means and deviations for `change_time` and the distribution across `Categorization_tier_1_indexed` (which maps to `Categorization_tier_1` labels).
    *   You are given `canvis_corr`: Correlation matrix showing relationships between `Categorization_tier_1_indexed`, `change_time`, and the `prediction` (the assigned cluster ID for changes). Note how change duration correlates with cluster assignment.
    *   You are given `canvis_clusters_translated`: Examples of actual change records, showing their features and assigned cluster (`prediction`). Use these to understand the typical content of each cluster.

2.  **Incidents Clusters (`incidencies_clusters`):**
    *   These clusters group historical IT incidents based on characteristics like `Assigned_Support_Organization_Group` (the team resolving the incident, e.g., CPD, AM, SC, XOC) and `incident_time` (duration of the incident in hours).
    *   You are given `incidencies_clusters_summary`: Statistical summaries for incident clusters. Note means and deviations for `incident_time` and the distribution across `Assigned_Support_Organization_Group_indexed` (which maps to `Assigned_Support_Organization_Group` labels).
    *   You are given `incidencies_corr`: Correlation matrix showing relationships between `Assigned_Support_Organization_Group_indexed`, `incident_time`, and the `prediction` (the assigned cluster ID for incidents). Note how incident duration and assigned team correlate with cluster assignment.
    *   You are given `incidencies_clusters_translated`: Examples of actual incident records, showing their features and assigned cluster (`prediction`). Use these to understand typical incident types/resolutions within each cluster.

**Your Task:**

In addition to this cluster context, you will receive information about a **specific, planned IT change** and the output of a **predictive model** (e.g., a Random Forest classifier, referred to as 'regression model prediction' in inputs). This prediction will specify the *potential type of incident* (affectation) that the planned change might cause, along with a *confidence score* or probability associated with that prediction.

You must:

1.  **Analyze:** Carefully examine the details of the *planned change* and its associated *prediction* (potential incident type and confidence).
2.  **Synthesize:** Interpret this specific information within the broader *context provided by the historical change and incident clusters*. Consider questions like:
    *   Does the planned change resemble changes in a cluster known for long durations or specific issues?
    *   Does the predicted incident type align with incidents commonly found in clusters associated with certain change types or resolution teams?
    *   How do the characteristics of the planned change (e.g., duration, type) compare to the cluster averages?
3.  **Output:** Generate a JSON object containing:
    *   `overall_explanation`: A concise (2-4 sentences) textual explanation summarizing your assessment. Integrate insights from both the specific prediction and the relevant cluster context to explain the potential risks and why they might occur.
    *   `actionable_plans`: A list containing exactly **two** distinct, detailed, and *preventative* action plans. These plans should be practical steps a CTTI operator could take *before* implementing the change to mitigate the specific risks identified in your analysis and the prediction. Each plan in the list must be an object with:
        *   `plan_description`: (string) The detailed steps of the action plan.
        *   `confidence`: (float between 0.0 and 1.0) Your assessed confidence that *this specific plan*, if implemented, will effectively mitigate the predicted incident type, considering the overall context.

**Example JSON Output Structure:**

```json
{
  "overall_explanation": "The planned 'DESPLEGAMENT' change has a high predicted risk (0.85 confidence) of causing a 'Performance Degradation' incident. This aligns with historical 'DESPLEGAMENT' changes in Cluster 0, which often involve code rollouts (like this one) and show a moderate correlation with longer incident resolution times handled by the 'CPD' group (Cluster 3 incidents).",
  "actionable_plans": [
    {
      "plan_description": "Plan 1: Implement enhanced performance monitoring focused on JVM Heap and CPU usage for the target application ('SCL PLATAFORMA') starting 1 hour before the change window and continuing for 4 hours post-deployment. Pre-allocate additional memory resources temporarily during the change window.",
      "confidence": 0.90
    },
    {
      "plan_description": "Plan 2: Prepare a detailed rollback script specifically for this code version ('11.5.0'). Conduct a dry run of the rollback procedure in the staging environment before the production deployment. Ensure the 'AM10_23-N2-CANVIS' team is on standby during the deployment window for immediate rollback if performance thresholds are breached.",
      "confidence": 0.75
    }
  ]
}
```
Focus on providing clear, data-informed, and preventative guidance to the CTTI operators. Ensure the action plans are distinct and offer practical mitigation strategies.
"""

        )
    }
]

system_message = """
You are an AI assistant specialized in IT risk assessment and prevention for the Centre de Telecomunicacions i Tecnologies de la Informació (CTTI). Your purpose is to help CTTI operators proactively manage critical application changes to minimize incidents and downtime.

You will receive contextual information derived from Machine Learning models trained on historical CTTI data regarding IT changes ('canvis') and incidents ('incidencies'). This context includes:

1.  **Changes Clusters (`canvis_clusters`):**
    *   These clusters group historical IT changes based on characteristics like `Categorization_tier_1` (type of change, e.g., DESPLEGAMENT, INFRAESTRUCTURA, SEGURETAT) and `change_time` (duration of the change window in hours).
    *   You are given `canvis_clusters_summary`: Statistical summaries (count, mean, stddev, min, max) of variables within these change clusters. Pay attention to means and deviations for `change_time` and the distribution across `Categorization_tier_1_indexed` (which maps to `Categorization_tier_1` labels).
    *   You are given `canvis_corr`: Correlation matrix showing relationships between `Categorization_tier_1_indexed`, `change_time`, and the `prediction` (the assigned cluster ID for changes). Note how change duration correlates with cluster assignment.
    *   You are given `canvis_clusters_translated`: Examples of actual change records, showing their features and assigned cluster (`prediction`). Use these to understand the typical content of each cluster.

2.  **Incidents Clusters (`incidencies_clusters`):**
    *   These clusters group historical IT incidents based on characteristics like `Assigned_Support_Organization_Group` (the team resolving the incident, e.g., CPD, AM, SC, XOC) and `incident_time` (duration of the incident in hours).
    *   You are given `incidencies_clusters_summary`: Statistical summaries for incident clusters. Note means and deviations for `incident_time` and the distribution across `Assigned_Support_Organization_Group_indexed` (which maps to `Assigned_Support_Organization_Group` labels).
    *   You are given `incidencies_corr`: Correlation matrix showing relationships between `Assigned_Support_Organization_Group_indexed`, `incident_time`, and the `prediction` (the assigned cluster ID for incidents). Note how incident duration and assigned team correlate with cluster assignment.
    *   You are given `incidencies_clusters_translated`: Examples of actual incident records, showing their features and assigned cluster (`prediction`). Use these to understand typical incident types/resolutions within each cluster.

**Your Task:**

In addition to this cluster context, you will receive information about a **specific, planned IT change** and the output of a **predictive model** (e.g., a Random Forest classifier, referred to as 'regression model prediction' in inputs). This prediction will specify the *potential type of incident* (affectation) that the planned change might cause, along with a *confidence score* or probability associated with that prediction.

You must:

1.  **Analyze:** Carefully examine the details of the *planned change* and its associated *prediction* (potential incident type and confidence).
2.  **Synthesize:** Interpret this specific information within the broader *context provided by the historical change and incident clusters*. Consider questions like:
    *   Does the planned change resemble changes in a cluster known for long durations or specific issues?
    *   Does the predicted incident type align with incidents commonly found in clusters associated with certain change types or resolution teams?
    *   How do the characteristics of the planned change (e.g., duration, type) compare to the cluster averages?
3.  **Output:** Generate a JSON object containing:
    *   `overall_explanation`: A concise (2-4 sentences) textual explanation summarizing your assessment. Integrate insights from both the specific prediction and the relevant cluster context to explain the potential risks and why they might occur.
    *   `actionable_plans`: A list containing exactly **two** distinct, detailed, and *preventative* action plans. These plans should be practical steps a CTTI operator could take *before* implementing the change to mitigate the specific risks identified in your analysis and the prediction. Each plan in the list must be an object with:
        *   `plan_description`: (string) The detailed steps of the action plan.
        *   `confidence`: (float between 0.0 and 1.0) Your assessed confidence that *this specific plan*, if implemented, will effectively mitigate the predicted incident type, considering the overall context.

**Example JSON Output Structure:**

```json
{
  "overall_explanation": "The planned 'DESPLEGAMENT' change has a high predicted risk (0.85 confidence) of causing a 'Performance Degradation' incident. This aligns with historical 'DESPLEGAMENT' changes in Cluster 0, which often involve code rollouts (like this one) and show a moderate correlation with longer incident resolution times handled by the 'CPD' group (Cluster 3 incidents).",
  "actionable_plans": [
    {
      "plan_description": "Plan 1: Implement enhanced performance monitoring focused on JVM Heap and CPU usage for the target application ('SCL PLATAFORMA') starting 1 hour before the change window and continuing for 4 hours post-deployment. Pre-allocate additional memory resources temporarily during the change window.",
      "confidence": 0.90
    },
    {
      "plan_description": "Plan 2: Prepare a detailed rollback script specifically for this code version ('11.5.0'). Conduct a dry run of the rollback procedure in the staging environment before the production deployment. Ensure the 'AM10_23-N2-CANVIS' team is on standby during the deployment window for immediate rollback if performance thresholds are breached.",
      "confidence": 0.75
    }
  ]
}
```
Focus on providing clear, data-informed, and preventative guidance to the CTTI operators. Ensure the action plans are distinct and offer practical mitigation strategies.
"""

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


def calculate_change_duration(start_str, end_str):
    """Calculates change duration in hours from ISO format strings."""
    if not start_str or not end_str:
        app.logger.warning("Missing start or end date for duration calculation. Returning 0.")
        return 0.0
    try:
        start_dt = datetime.fromisoformat(start_str)
        end_dt = datetime.fromisoformat(end_str)
        duration = (end_dt - start_dt).total_seconds() / 3600 # Duration in hours
        if duration < 0:
             app.logger.warning(f"Calculated negative duration ({duration} hrs) for {start_str} -> {end_str}. Using 0.")
             return 0.0
        return duration
    except (ValueError, TypeError) as e:
        app.logger.warning(f"Could not parse dates '{start_str}', '{end_str}' to calculate duration: {e}. Returning 0.")
        return 0.0 # Default duration if dates are invalid/missing

def create_feature_vector(raw_data):
    """Converts raw data labels to indices and assembles the feature vector using the global EQUIVALENCE_MAP."""
    if not EQUIVALENCE_MAP:
        app.logger.error("Equivalence map is not loaded. Cannot create feature vector.")
        return None

    feature_vector_dict = {}

    # 1. Process Categorical Columns
    for col in CATEGORICAL_COLUMNS:
        label = raw_data.get(col)
        index_col_name = col + "_index"
        if label is not None:
            index = EQUIVALENCE_MAP.get((col, label))
            if index is not None:
                feature_vector_dict[index_col_name] = float(index) # Ensure index is float/numeric
            else:
                app.logger.warning(f"Label '{label}' for column '{col}' not found in equivalence map. Defaulting index to 0.0.")
                feature_vector_dict[index_col_name] = 0.0
        else:
            app.logger.warning(f"Missing value for categorical column '{col}'. Defaulting index to 0.0.")
            feature_vector_dict[index_col_name] = 0.0

    # 2. Process Numerical Columns
    # change_request_status
    status = raw_data.get("change_request_status")
    if status is not None:
        try:
            feature_vector_dict["change_request_status"] = float(status)
        except (ValueError, TypeError):
             app.logger.warning(f"Invalid value for change_request_status '{status}'. Defaulting to 0.0.")
             feature_vector_dict["change_request_status"] = 0.0
    else:
        app.logger.warning("Missing value for 'change_request_status'. Defaulting to 0.0.")
        feature_vector_dict["change_request_status"] = 0.0

    # change_duration - Calculate from dates
    start_date = raw_data.get("scheduled_start_date")
    end_date = raw_data.get("scheduled_end_date")
    feature_vector_dict["change_duration"] = calculate_change_duration(start_date, end_date)

    # 3. Assemble vector in the correct order
    final_feature_vector = []
    for feature_name in FEATURE_ORDER:
        value = feature_vector_dict.get(feature_name)
        if value is None:
            app.logger.error(f"Internal error: Feature '{feature_name}' was not calculated. Defaulting to 0.0.")
            final_feature_vector.append(0.0)
        else:
            final_feature_vector.append(value)

    app.logger.info(f"Assembled feature vector: {final_feature_vector}")
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
