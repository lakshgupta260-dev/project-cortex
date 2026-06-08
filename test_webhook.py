import requests

url = "https://jarring-stingily-crummiest.ngrok-free.dev/webhook"

payload = {
    "entry": [
        {
            "changes": [
                {
                    "value": {
                        "messages": [
                            {
                                "from": "919625433606",
                                "text": {
                                    "body": "hi"
                                }
                            }
                        ]
                    }
                }
            ]
        }
    ]
}

r = requests.post(url, json=payload)

print("STATUS:", r.status_code)
print("TEXT:", r.text)