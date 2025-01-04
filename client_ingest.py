import requests

url = "http://0.0.0.0:80/ingest"
files = [
    ('files', ('Relativity.pdf', open('./PDFs/Relativity.pdf', 'rb'), 'application/pdf'))
]
data = {
    'chunk_overlap': 100,
    'chunk_size': 1000,
    'request_id': 'req_2024_12_26_AA',
    'user_id': 'user123',
    'embedding_model_name': 'thenlper/gte-small'
}

response = requests.post(url, files=files, data=data)
print(response.json())