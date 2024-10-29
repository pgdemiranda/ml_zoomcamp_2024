import json
import requests

json_file = 'customer.json'

with open(json_file, 'r') as f:
    customer_data = json.load(f)

url = 'http://localhost:9696/predict'

response = requests.post(url, json=customer_data)

print(response.json())
