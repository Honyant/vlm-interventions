import os
import requests

# Input
input_text = "Hello, how are you?"

# Message payload
messages = [
            {"role": "user", "content": input_text}
            ]

# API payload
payload = {
            "model": "gpt-4",
                "messages": messages,
                    "max_tokens": 100
                    }

# Headers
headers = {
            "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
                }

# Make the request
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
print(response.json()["choices"][0]["message"]["content"])

