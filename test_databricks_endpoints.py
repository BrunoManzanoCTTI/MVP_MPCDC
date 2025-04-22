import requests
import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Get Databricks token and endpoint URLs from environment variables
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
MPCDC_REGRESSION_ENDPOINT = os.getenv("MPCDC_REGRESSION_ENDPOINT", "https://adb-2869758279805397.17.azuredatabricks.net/serving-endpoints/New_MPCDC_Regression_Endpoint/invocations")
EQUIVALENCE_CSV_PATH = "AI_Failure_Prediction_and_Prevention_for_CTTI.csv" # Path to your equivalence CSV

# --- Configuration ---
if not DATABRICKS_TOKEN:
    logging.error("DATABRICKS_TOKEN environment variable not set. Please set it in the .env file.")
    exit(1)

if not MPCDC_REGRESSION_ENDPOINT:
    logging.error("MPCDC_REGRESSION_ENDPOINT URL is not defined. Check environment variables or defaults.")
    exit(1)

HEADERS = {'Authorization': f'Bearer {DATABRICKS_TOKEN}', 'Content-Type': 'application/json'}

# Define the order of features expected by the model endpoint
# Matches the VectorAssembler inputCols: categorical indices + numerical
FEATURE_ORDER = [
    "f01_chr_serviceid_index", "serviceci_index", "ASORG_index", "ASGRP_index",
    "categorization_tier_1_index", "categorization_tier_2_index", "categorization_tier_3_index",
    "product_cat_tier_1_index", "product_cat_tier_2_index", "product_cat_tier_3_index",
    "change_request_status", "change_duration"
]

CATEGORICAL_COLUMNS = [col.replace('_index', '') for col in FEATURE_ORDER if col.endswith('_index')]
NUMERICAL_COLUMNS = [col for col in FEATURE_ORDER if not col.endswith('_index')]

# --- Sample Input Data (Using user provided sample) ---
sample_change_data = {
    "infrastructure_change_id": "CRQ000000866955",
    "submit_date": "2024-09-26T14:00:00", # Keeping previous placeholder
    "scheduled_start_date": "2024-09-30T12:00:00", # Keeping previous placeholder
    "scheduled_end_date": "2024-09-30T16:00:00", # Keeping previous placeholder
    "f01_chr_serviceid": "ST.APP.00673",
    "serviceci": "SCL PLATAFORMA, ATENCIO PERSONALITZADA I CERCADORS",
    "ASORG": "AM16_23",
    "ASGRP": "AM16_23-N2-CANVIS",
    "categorization_tier_1": "DESPLEGAMENT",
    "categorization_tier_2": "CODI",
    "categorization_tier_3": "CORRECTIU",
    "product_cat_tier_1": "APLICACIONS",
    "product_cat_tier_2": "AM16-CPD3",
    "product_cat_tier_3": "-", # Example with a common placeholder
    "change_request_status": 11, # Numerical feature
    # Num_Incidencies is not used in the feature vector for the regression model
    # f01_chr_tipoafectacion is the target variable, not an input feature
}


def load_equivalence_map(csv_path):
    """Loads the equivalence CSV into a lookup dictionary."""
    logging.info(f"Loading equivalence map from: {csv_path}")
    try:
        # IMPORTANT: Assuming the CSV is not excessively large for memory
        # If it's huge, a database or more optimized lookup might be needed.
        df_equiv = pd.read_csv(csv_path)
        # Create a multi-index for faster lookups if needed, or use a dictionary
        # Dictionary approach: {(column, label): index}
        equiv_map = {}
        for _, row in df_equiv.iterrows():
            equiv_map[(row['Column'], row['Label'])] = row['Index']
        logging.info(f"Successfully loaded {len(equiv_map)} mappings.")
        return equiv_map
    except FileNotFoundError:
        logging.error(f"Equivalence CSV not found at: {csv_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading or processing equivalence CSV: {e}")
        return None

def calculate_change_duration(start_str, end_str):
    """Calculates change duration in hours from ISO format strings."""
    try:
        # Use ISO format directly if available, otherwise adapt parsing
        start_dt = datetime.fromisoformat(start_str)
        end_dt = datetime.fromisoformat(end_str)
        duration = (end_dt - start_dt).total_seconds() / 3600 # Duration in hours
        if duration < 0:
             logging.warning(f"Calculated negative duration ({duration} hrs) for {start_str} -> {end_str}. Using 0.")
             return 0.0
        return duration
    except (ValueError, TypeError) as e:
        logging.warning(f"Could not parse dates '{start_str}', '{end_str}' to calculate duration: {e}. Returning 0.")
        return 0.0 # Default duration if dates are invalid/missing

def create_feature_vector(raw_data, equiv_map):
    """Converts raw data labels to indices and assembles the feature vector."""
    if not equiv_map:
        logging.error("Equivalence map is not loaded. Cannot create feature vector.")
        return None

    feature_vector_dict = {}

    # 1. Process Categorical Columns
    for col in CATEGORICAL_COLUMNS:
        label = raw_data.get(col)
        index_col_name = col + "_index"
        if label is not None:
            # Lookup in the equivalence map
            index = equiv_map.get((col, label))
            if index is not None:
                feature_vector_dict[index_col_name] = float(index) # Ensure index is float/numeric
            else:
                # Handle unknown labels - default to 0.0 based on 'skip' behavior?
                # Or use a specific value like -1.0 if the model handles it.
                # Using 0.0 for now, assuming it aligns with how 'skip' might effectively work
                # during training (assigning to the most frequent category's index, often 0).
                logging.warning(f"Label '{label}' for column '{col}' not found in equivalence map. Defaulting index to 0.0.")
                feature_vector_dict[index_col_name] = 0.0
        else:
            # Handle missing categorical value in input
            logging.warning(f"Missing value for categorical column '{col}'. Defaulting index to 0.0.")
            feature_vector_dict[index_col_name] = 0.0

    # 2. Process Numerical Columns
    # change_request_status
    status = raw_data.get("change_request_status")
    if status is not None:
        try:
            feature_vector_dict["change_request_status"] = float(status)
        except (ValueError, TypeError):
             logging.warning(f"Invalid value for change_request_status '{status}'. Defaulting to 0.0.")
             feature_vector_dict["change_request_status"] = 0.0
    else:
        logging.warning("Missing value for 'change_request_status'. Defaulting to 0.0.")
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
            # This shouldn't happen if all features are processed above, but as a safeguard
            logging.error(f"Internal error: Feature '{feature_name}' was not calculated. Defaulting to 0.0.")
            final_feature_vector.append(0.0)
        else:
            final_feature_vector.append(value)

    logging.info(f"Assembled feature vector: {final_feature_vector}")
    return final_feature_vector


def call_databricks_endpoint(endpoint_url, payload):
    """Helper function to call a Databricks endpoint."""
    try:
        # Using orjson if available for potentially faster NaN/Inf handling, otherwise standard json
        try:
            import orjson
            # Need to handle potential numpy types before orjson serialization
            def default_serializer(obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    # Handle NaN specifically for orjson
                    if np.isnan(obj):
                        return None # Or another JSON-serializable representation like 'NaN' string
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)): # Handle numpy arrays if they appear
                    return obj.tolist()
                elif isinstance(obj, (np.bool_)):
                    return bool(obj)
                elif isinstance(obj, (np.void)):
                    return None
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
            payload_json = orjson.dumps(payload, default=default_serializer).decode('utf-8')

        except ImportError:
            # Standard json, handle potential NaN/Inf if necessary
            def default_serializer_std(obj):
                 if isinstance(obj, np.integer):
                     return int(obj)
                 elif isinstance(obj, np.floating):
                     return float(obj) if not np.isnan(obj) else None # Convert NaN to None for standard JSON
                 elif isinstance(obj, np.ndarray):
                     return obj.tolist()
                 elif isinstance(obj, np.bool_):
                     return bool(obj)
                 raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
            payload_json = json.dumps(payload, default=default_serializer_std, allow_nan=False) # Be strict with NaN/Inf

        response = requests.post(endpoint_url, headers=HEADERS, data=payload_json)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling endpoint {endpoint_url}: {e}")
        if e.response is not None:
            logging.error(f"Response status code: {e.response.status_code}")
            logging.error(f"Response text: {e.response.text}")
        return None
    except (TypeError, ValueError) as e: # Catch JSON encoding errors
        logging.error(f"Error encoding payload to JSON: {e}")
        logging.error(f"Payload causing error (sample): {str(payload)[:500]}...") # Log sample of payload
        return None


def main():
    logging.info("--- Starting Databricks Endpoint Test (Local Preprocessing) ---")
    logging.info(f"Using Regression Endpoint: {MPCDC_REGRESSION_ENDPOINT}")

    # --- Step 1: Load Equivalence Map ---
    logging.info("\n--- Step 1: Loading Equivalence Map ---")
    equivalence_map = load_equivalence_map(EQUIVALENCE_CSV_PATH)
    if not equivalence_map:
        logging.error("Failed to load equivalence map. Exiting.")
        return

    # --- Step 2: Create Feature Vector Locally ---
    logging.info("\n--- Step 2: Creating Feature Vector Locally ---")
    logging.info(f"Raw input data: {json.dumps(sample_change_data, indent=2)}")
    feature_vector = create_feature_vector(sample_change_data, equivalence_map)

    if not feature_vector:
        logging.error("Failed to create feature vector. Exiting.")
        return

    # --- Step 3: Prepare Payload and Call Regression Model ---
    logging.info("\n--- Step 3: Calling Regression Model ---")
    try:
        # The regression model endpoint expects a DataFrame format with a 'features' column
        # The 'features' column should contain the list (vector) of features.
        regression_payload_df = pd.DataFrame({'features': [feature_vector]}) # Wrap the vector in a list

        # Convert to dataframe_split format expected by the endpoint
        regression_payload_dict_raw = regression_payload_df.to_dict(orient='split')

        # Ensure data part is correctly formatted: list of lists, where each inner list contains the feature vector
        # Example: {'columns': ['features'], 'data': [ [[feature1, feature2, ...]] ]}
        # The to_dict(orient='split') might produce {'data': [[vector]]}, which is correct.
        # Let's double-check and adjust if necessary. Pandas usually handles this correctly for a single column of lists/vectors.
        # No, to_dict(orient='split') on a DataFrame with a list/vector column creates:
        # {'columns': ['features'], 'data': [ [[item1, item2, ...]] ]} - This is what we need.

        regression_payload = {'dataframe_split': regression_payload_dict_raw}

        # Remove the 'index' key if present, as it's not needed for inference
        if 'index' in regression_payload['dataframe_split']:
            del regression_payload['dataframe_split']['index']

        logging.info(f"Payload for Regression Model: {json.dumps(regression_payload, indent=2)}")
        regression_result = call_databricks_endpoint(MPCDC_REGRESSION_ENDPOINT, regression_payload)

        if not regression_result:
            logging.error("Failed to get response from regression model.")
            return

        logging.info(f"Regression Model Response: {json.dumps(regression_result, indent=2)}")

        # --- Step 4: Interpret Final Prediction ---
        logging.info("\n--- Step 4: Final Prediction ---")
        # (Keep the existing prediction interpretation logic)
        final_prediction_value = None
        if 'predictions' in regression_result and isinstance(regression_result['predictions'], list) and regression_result['predictions']:
             pred_output = regression_result['predictions'][0]
             # The regression endpoint might return the prediction directly as a number
             if isinstance(pred_output, (int, float)):
                  final_prediction_value = float(pred_output)
             # Or it might return a dictionary containing the prediction
             elif isinstance(pred_output, dict) and 'prediction' in pred_output:
                  final_prediction_value = pred_output['prediction']
             else:
                 logging.warning(f"Unexpected prediction format in regression response: {pred_output}")

        if final_prediction_value is not None:
             # Map numeric prediction back to the correct f01_chr_tipoafectacion label
             type_mapping = {
                 0.0: "SENSE TALL DE SERVEI",
                 1.0: "TALL DE SERVEI",
                 2.0: "DEGRADACIO"
             }
             predicted_label = type_mapping.get(final_prediction_value, f"UNKNOWN_CODE_{final_prediction_value}")
             logging.info(f"Predicted Label: {predicted_label} (Raw value: {final_prediction_value})")
        else:
             logging.warning("Could not extract final prediction from regression model response.")
             logging.info(f"Full regression response for debugging: {json.dumps(regression_result, indent=2)}")

    except (ValueError, KeyError, IndexError, TypeError) as e:
        logging.error(f"Error preparing data for or calling regression model: {e}")
        # Log the feature vector if it exists, as it's crucial for debugging Step 3
        if 'feature_vector' in locals():
            logging.error(f"Feature vector causing issue (if available): {feature_vector}")

    logging.info("\n--- Test Complete ---")

if __name__ == "__main__":
    main()
