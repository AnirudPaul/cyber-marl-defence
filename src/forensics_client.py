"""
Example client to POST a sample to /explain
"""
import requests
import numpy as np

SERVER = "http://127.0.0.1:5001"
URL = SERVER + "/explain"

def send_random_sample(n_features=15, target=None):
    sample = np.random.randn(n_features).tolist()
    payload = {
        "id": "test_sample_001",
        "features": sample
    }
    if target is not None:
        payload["target"] = int(target)
    r = requests.post(URL, json=payload, timeout=30)
    print("HTTP:", r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text)

if __name__ == "__main__":
    send_random_sample(n_features=15, target=2)
