import requests
import json
import time
import argparse

BASE_URL = "http://localhost:8000"

def test_all(model, api_key):
    # Wait for server to start
    time.sleep(2)
    
    # 1. Configure (Test OpenAI config)
    print(f"Configuring {model}...")
    res = requests.post(f"{BASE_URL}/configure", json={
        "provider": "openai",
        "model": model,
        "api_key": api_key
    })
    print(res.json())
    assert res.status_code == 200
    
    # 2. Register
    print("Registering...")
    res = requests.post(f"{BASE_URL}/register", json={
        "name": "qa",
        "signature": "question -> answer",
        "instructions": "Answer the question concisely and accurately."
    })
    print(res.json())
    assert res.status_code == 200

    # 3. Predict (Zero-shot)
    print("Predicting (Zero-shot)...")
    res = requests.post(f"{BASE_URL}/predict", json={
        "signature_name": "qa",
        "inputs": {"question": "What is the capital of France?"},
        "module_type": "ChainOfThought"
    })
    print("Response:", res.json())
    assert res.status_code == 200
    assert "answer" in res.json()
    print(f"Q: What is the capital of France? A: {res.json()['answer']}")

    # 4. Optimize
    print("Optimizing...")
    # Real training data
    train_data = [
        {"question": "What is the capital of Germany?", "answer": "Berlin"},
        {"question": "What is the capital of Italy?", "answer": "Rome"},
        {"question": "What is the capital of Spain?", "answer": "Madrid"}
    ]
    res = requests.post(f"{BASE_URL}/optimize", json={
        "signature_name": "qa",
        "train_data": train_data,
        "metric": "exact_match",
        "max_bootstraps": 2
    })
    print(res.json())
    assert res.status_code == 200
    module_id = res.json()["module_id"]

    # 5. Predict with optimized module
    print(f"Predicting with optimized module ({module_id})...")
    res = requests.post(f"{BASE_URL}/predict", json={
        "signature_name": "qa",
        "inputs": {"question": "What is the capital of Portugal?"},
        "compiled_module_id": module_id
    })
    print("Response:", res.json())
    assert res.status_code == 200
    print(f"Q: What is the capital of Portugal? A: {res.json()['answer']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DSPy Proxy Server")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--api-key", type=str, required=True, help="API Key")
    args = parser.parse_args()
    
    test_all(args.model, args.api_key)
