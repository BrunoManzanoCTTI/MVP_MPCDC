#!/usr/bin/env python
"""
Example script demonstrating how to use the classify_change function directly.
This is useful for integrating the function into other Python applications.
"""

import sys
import os
import json
from datetime import datetime, timedelta

# Add the current directory to the path so we can import from app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the classify_change function from app.py
from app import classify_change

# Sample change data
today = datetime.now()
start_date = today + timedelta(days=1)
end_date = today + timedelta(days=2)

sample_change = {
    "infrastructure_change_id": "CHG123456",
    "submit_date": today,  # We can pass datetime objects directly
    "scheduled_start_date": start_date,
    "scheduled_end_date": end_date,
    "f01_chr_serviceid": "SVC001",
    "serviceci": "CI001",
    "ASORG": "IT",
    "ASGRP": "Infrastructure",
    "categorization_tier_1": "INFRAESTRUCTURA",
    "categorization_tier_2": "Server",
    "categorization_tier_3": "Configuration",
    "product_cat_tier_1": "Hardware",
    "product_cat_tier_2": "Server",
    "product_cat_tier_3": "Configuration",
    "change_request_status": 2,
    "Num_Incidencies": "INC000009582237"
}

# Call the classify_change function
result = classify_change(sample_change)

# Print the result
print("Classification Result:")
print(json.dumps(result, indent=2))

# Example of how to use the result in your application
if result.get("status") == "success":
    prediction = result.get("prediction")
    print(f"\nPrediction: {prediction}")
    
    # Example decision logic based on prediction
    if isinstance(prediction, (int, float)) and prediction > 0.7:
        print("High risk change detected! Additional approval required.")
    elif isinstance(prediction, str) and "high" in prediction.lower():
        print("High risk change detected! Additional approval required.")
    else:
        print("Standard change. Proceed with normal approval process.")
else:
    print(f"\nError: {result.get('message')}")
