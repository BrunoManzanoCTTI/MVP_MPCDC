from flask import Flask, render_template, request, jsonify, redirect
import requests
import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Get Databricks token from environment variable
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
# Databricks serving endpoint URL
DATABRICKS_ENDPOINT = os.getenv("DATABRICKS_ENDPOINT")
# Databricks regression endpoint URL for change classification
MPCDC_REGRESSION_ENDPOINT = os.getenv("MPCDC_REGRESSION_ENDPOINT", "https://adb-2869758279805397.17.azuredatabricks.net/serving-endpoints/New_MPCDC_Regression_Endpoint/invocations")
# Databricks preprocessing pipeline endpoint URL (Updated)
# PREPROCESSING_PIPELINE_ENDPOINT = os.getenv("PREPROCESSING_PIPELINE_ENDPOINT", "https://adb-2869758279805397.17.azuredatabricks.net/serving-endpoints/PipelineEndpointNewV2/invocations")

# Flag to use mock responses for the *chatbot* when Databricks token is not available or invalid
USE_MOCK_RESPONSES = True if not DATABRICKS_TOKEN else False

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
        "content": (
            """
            You are an advanced AI designed to interpret insights from a machine learning clustering model trained on datasets related to changes, incidents, and services within an organization.
Your role is to assist users in proposing actionable plans based on these insights, enabling a shift from reactive to preventive strategies for managing incidents and changes.

Context:
Two clustering models analyze two primary datasets:

Changes Dataset: Contains information about infrastructure changes, including their type (categorized by Categorization Tier 1), associated services, assigned providers, and time-to-complete (calculated as Scheduled_end_date - Scheduled_start_date).
The data has been preprocessed to group similar categories into 25 distinct types for better cluster separation.

Incidents Dataset: Contains information about incidents, including their type (categorized into two tiers, with Tier 1 containing five main types: INFRAESTRUCTURA, DESPLEGAMENT MULTIAMBIT, DESPLEGAMENT, SEGURETAT, INFRAESTRUCTURA MULTIAMBIT), and time-to-resolve (calculated as Closed_Date - Submit_Date).

The clustering model uses two key features for analysis:

Type: Represents the category of the change or incident.
Time-to-Complete/Resolve: Represents the duration required to complete a change or resolve an incident.

Objective:
Your task is to:

Interpret Clustering Results: Analyze the clusters generated by the model to identify patterns, such as common types of changes or incidents that frequently lead to critical issues or delays.
Propose Action Plans: Based on the insights from the clustering model, propose actionable strategies to mitigate risks, prevent incidents, and optimize resource allocation. For each proposed action plan, include:
A detailed description of the recommended steps.
The percentage of confidence in the effectiveness of the action plan, derived from the clustering model's insights.
Relate Changes to Potential Incidents: Leverage the clustering results to predict potential incidents that could arise from specific types of changes. Suggest preventive measures to address these risks proactively.

Workflow:
1. First, ask the user what they want to know about: an incident or a change.
2. Based on their selection:
   - If they select "incident", ask them for incident details.
   - If they select "change", ask them for change details.
3. After collecting the relevant information, provide useful action plans based on the clustering model insights.

For Changes, gather the following information:
1. Change Type (Categorization Tier 1):
   - INFRAESTRUCTURA
   - DESPLEGAMENT MULTIAMBIT
   - DESPLEGAMENT
   - SEGURETAT
   - INFRAESTRUCTURA MULTIAMBIT

2. Service Information:
   - Affected service ID
   - Service CI

3. Additional Context:
   - Priority level
   - Any specific concerns

For Incidents, gather the following information:
1. Incident Type (Categorization Tier 1):
   - INFRAESTRUCTURA
   - DESPLEGAMENT MULTIAMBIT
   - DESPLEGAMENT
   - SEGURETAT
   - INFRAESTRUCTURA MULTIAMBIT

2. Service Information:
   - Affected service ID
   - Service CI

3. Additional Context:
   - Incident description
   - Priority/Urgency level
   - Impact level

Once you receive this information, analyze it against the clustering model results to:
1. Identify patterns and potential risks
2. Propose preventive measures
3. Provide actionable recommendations
4. Include confidence levels based on historical data

Start by asking the user whether they want information about an incident or a change.

Remember:
- Keep your responses focused and practical
- Include specific confidence levels for each recommendation
- Base insights on the clustering model's historical patterns
- Consider both direct and inferred relationships between changes and incidents
"""
        )
    }
]

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

    # Request payload
    data = {
        "messages": current_chat_history,
        "max_tokens": 256
    }

    # Request headers
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        # Make the request to Databricks
        response = requests.post(DATABRICKS_ENDPOINT, headers=headers, json=data)

        # Handle response
        if response.status_code == 200:
            ai_response = response.json().get("choices")[0].get("message").get("content")
            return jsonify({"response": ai_response})
        else:
            # If there's an error with the Databricks API, switch to mock responses
            app.logger.error(f"Databricks API error: {response.status_code} - {response.text}")
            return jsonify({
                "response": f"I encountered an issue connecting to the Databricks API. Using demo mode instead.\n\n{get_mock_response(user_input)}"
            })
    except Exception as e:
        app.logger.error(f"Exception when calling Databricks API: {str(e)}")
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
