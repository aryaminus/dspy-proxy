import requests
import json
import time
import argparse

def test_all(base_url, model, api_key, provider):
    print(f"Testing against {base_url}")
    
    # 1. Configure (Test OpenAI config)
    print(f"Configuring {model}...")
    try:
        # Increased timeout for cold starts on free tier
        res = requests.post(f"{base_url}/configure", json={
            "provider": provider,
            "model": model,
            "api_key": api_key
        }, timeout=60) 
    except requests.exceptions.Timeout:
        print("Request timed out. The server might be waking up (cold start). Retrying...")
        res = requests.post(f"{base_url}/configure", json={
            "provider": provider,
            "model": model,
            "api_key": api_key
        }, timeout=60)
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to {base_url}. Is the server running?")
        return

    print(res.json())
    assert res.status_code == 200
    
    # 2. Register
    print("Registering...")
    res = requests.post(f"{base_url}/register", json={
        "name": "qa",
        "signature": "question -> answer",
        "instructions": "Answer the question concisely and accurately."
    })
    print(res.json())
    assert res.status_code == 200

    # 3. Predict (Zero-shot)
    print("Predicting (Zero-shot)...")
    res = requests.post(f"{base_url}/predict", json={
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
    res = requests.post(f"{base_url}/optimize", json={
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
    res = requests.post(f"{base_url}/predict", json={
        "signature_name": "qa",
        "inputs": {"question": "What is the capital of Portugal?"},
        "compiled_module_id": module_id
    })
    print("Response:", res.json())
    assert res.status_code == 200
    print(f"Q: What is the capital of Portugal? A: {res.json()['answer']}")

    # 6. Evaluate
    print("Evaluating...")
    test_data = [
        {"question": "What is 5+5?", "answer": "10"},
        {"question": "What is the capital of France?", "answer": "Paris"}
    ]
    res = requests.post(f"{base_url}/evaluate", json={
        "signature_name": "qa",
        "test_data": test_data,
        "metric": "exact_match",
        "compiled_module_id": module_id
    })
    print("Evaluation Response:", res.json())
    assert res.status_code == 200
    print(f"Score: {res.json()['score']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DSPy Proxy Server")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base URL of the server")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--api-key", type=str, required=True, help="API Key")
    parser.add_argument("--provider", type=str, default="openai", help="Provider to use")
    args = parser.parse_args()
    
    test_all(args.url, args.model, args.api_key, args.provider)
