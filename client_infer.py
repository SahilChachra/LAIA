import requests

url = "http://127.0.0.1:80/generate"  # Replace with the actual FastAPI endpoint
payload = {
    "question": "Tell me about String theory",
    "use_pdf_rag": False,
    "use_web_search": True,
    "request_id": "1bdeoka89-3fa1s212",
    "user_id": "user123"
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

# Check the response
print(response.json())
