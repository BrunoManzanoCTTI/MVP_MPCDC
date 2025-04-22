#!/usr/bin/env python
"""
Test script for the classify_change endpoint.
This script demonstrates how to call the endpoint with sample data.
"""

import requests
import json
from datetime import datetime, timedelta

# URL of the classify_change endpoint
URL = "http://localhost:5000/mpcdc/classify_change"

# Sample change data
today = datetime.now()
start_date = today + timedelta(days=1)
end_date = today + timedelta(days=2)

sample_change = {
    "infrastructure_change_id": "CHG123456",
    "submit_date": today.strftime('%d/%m/%Y %H:%M:%S'),
    "scheduled_start_date": start_date.strftime('%d/%m/%Y %H:%M:%S'),
    "scheduled_end_date": end_date.strftime('%d/%m/%Y %H:%M:%S'),
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

# Make the request
headers = {
    "Content-Type": "application/json"
}

response = requests.post(URL, headers=headers, json=sample_change)

# Print the response
print(f"Status Code: {response.status_code}")
print("Response:")
print(json.dumps(response.json(), indent=2))
