# DSPy Proxy Server

A proxy server for [DSPy](https://github.com/stanfordnlp/dspy) that allows you to register signatures, execute modules, and run optimization procedures via a REST API. This enables using DSPy from languages other than Python.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/aryaminus/dspy-proxy)

## Features

- **Register Signatures**: Define input/output signatures dynamically.
- **Execute Modules**: Run `Predict` or `ChainOfThought` modules.
- **Optimize**: Compile modules using `BootstrapFewShot` optimizer.
- **Configure LM**: Set up the Language Model (OpenAI supported).

## Installation

1. Clone the repository.
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

Start the server:
```bash
python main.py
```

The server runs on `http://0.0.0.0:8000`.

## Deployment

### Option 1: One-Click Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/aryaminus/dspy-proxy)

### Option 2: Manual Deployment

This project is configured for deployment on [Render](https://render.com).

1.  Fork this repository.
2.  Create a new **Web Service** on Render.
3.  Connect your repository.
4.  Render will automatically detect the `render.yaml` configuration.
5.  Add your `OPENAI_API_KEY` in the Environment Variables settings on Render.

## API Endpoints

### 1. Configure LM
`POST /configure`
```json
{
  "provider": "openai",
  "model": "gpt-4o-mini",
  "api_key": "sk-..."
}
```
Supported providers: `openai`, `anthropic`, `google`, `azure`, etc. (any provider supported by `litellm`).
If `api_key` is omitted, the server will look for environment variables like `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.

### 2. Register Signature
`POST /register`
```json
{
  "name": "qa",
  "signature": "question -> answer",
  "instructions": "Answer the question concisely."
}
```

### 3. Predict
`POST /predict`
```json
{
  "signature_name": "qa",
  "inputs": {
    "question": "What is the capital of France?"
  },
  "module_type": "ChainOfThought"
}
```

### 4. Optimize
`POST /optimize`
```json
{
  "signature_name": "qa",
  "train_data": [
    {"question": "What is 1+1?", "answer": "2"},
    {"question": "What is 2+2?", "answer": "4"}
  ],
  "metric": "exact_match",
  "optimizer": "BootstrapFewShot",
  "max_bootstraps": 4
}
```
Returns a `module_id`.

### 5. Predict with Optimized Module
`POST /predict`
```json
{
  "signature_name": "qa",
  "inputs": {
    "question": "What is 3+3?"
  },
  "compiled_module_id": "qa_opt_0"
}
```
