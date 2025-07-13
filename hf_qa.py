
import requests

API_URL = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-cased-distilled-squad"
HEADERS = {"Authorization": "Bearer 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"}

def query_huggingface_qa(question: str, context: str):
    payload = {
        "inputs": {
            "question": question,
            "context": context
        }
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()


