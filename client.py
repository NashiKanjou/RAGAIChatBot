import requests
import json


json_data = {}

question = input("Question: ")
db = input("Database to search (press enter for default): " or None)
model = input("Model name (press enter for default): " or None)

json_data['query'] = question
if(db):
    json_data['key'] = db
if(db):
    json_data['model'] = model
    
response = requests.post('http://localhost:8763', json=json_data)
data = response.json()
resp = data["response"]
for r in resp:
    print(r)