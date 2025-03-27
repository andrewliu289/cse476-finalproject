import requests

url = "http://127.0.0.1:8000/chat"
prompt = "Write a 100 word short story"

response = requests.post(url, json={"prompt": prompt})
print("Response:", response.json()["response"])
