# Change Classification API

This document describes how to use the change classification functionality in the MPCDC application.

## Overview

The change classification feature uses a machine learning model deployed on Azure Databricks to predict the risk level of infrastructure changes. The model analyzes various attributes of a change and provides predictions about its potential impact.

## API Endpoint

### POST /mpcdc/classify_change

This endpoint accepts change data in JSON format and returns a classification result.

**Request Format:**

```json
{
  "infrastructure_change_id": "CHG123456",
  "submit_date": "2023-03-31",
  "scheduled_start_date": "2023-04-01",
  "scheduled_end_date": "2023-04-02",
  "f01_chr_serviceid": "SVC001",
  "serviceci": "CI001",
  "ASORG": "IT",
  "ASGRP": "Infrastructure",
  "detailed_description": "Update server configurations to improve performance",
  "categorization_tier_1": "INFRAESTRUCTURA",
  "categorization_tier_2": "Server",
  "categorization_tier_3": "Configuration",
  "product_cat_tier_1": "Hardware",
  "product_cat_tier_2": "Server",
  "product_cat_tier_3": "Configuration",
  "change_request_status": "Scheduled",
  "Num_Incidencies": 0,
  "f01_chr_tipoafectacion": "None"
}
```

**Required Fields:**
- `infrastructure_change_id`
- `submit_date`
- `scheduled_start_date`
- `scheduled_end_date`
- `f01_chr_serviceid`
- `serviceci`

**Response Format (Success):**

```json
{
  "prediction": "Medium Risk",
  "raw_response": {
    "predictions": ["Medium Risk"]
  },
  "status": "success"
}
```

**Response Format (Error):**

```json
{
  "status": "error",
  "message": "Error from Databricks Regression API: 400",
  "details": "Error details from the API"
}
```

## Direct Function Usage

You can also use the `classify_change` function directly in your Python code:

```python
from app import classify_change

# Prepare change data
change_data = {
    "infrastructure_change_id": "CHG123456",
    "submit_date": "2023-03-31",
    # Add other required fields...
}

# Call the function
result = classify_change(change_data)

# Process the result
if result.get("status") == "success":
    prediction = result.get("prediction")
    # Handle the prediction...
else:
    # Handle error...
    error_message = result.get("message")
```

## Example Scripts

Two example scripts are provided to demonstrate how to use the change classification functionality:

1. `test_classify_change.py` - Shows how to call the API endpoint
2. `example_direct_usage.py` - Shows how to use the function directly in Python code

## Demo Mode

When the application is running in demo mode (no valid Databricks token), the classification function will return mock responses for testing purposes.

## Azure Databricks Endpoint

The classification uses the following Azure Databricks endpoint:
```
https://adb-2869758279805397.17.azuredatabricks.net/serving-endpoints/MPCDC_Regression_Endpoint/invocations
```

To use the actual endpoint, make sure to set the `DATABRICKS_TOKEN` environment variable in your `.env` file.
